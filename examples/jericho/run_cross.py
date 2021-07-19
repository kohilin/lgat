import logging
import glob
import sys
import os
import tqdm
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser
from pfrl.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer

from exp import create_md_table_from_dic

sys.path.append('../../')
from lgat.jericho import Agent, RNNEncoder, JerichoBatchEnv, JerichoMultiGameBatchEnv
from lgat.generator import TemplateAction, LGAT
from lgat.action import TDQNTemplateCollection
from lgat.action.masker import create_masker_params_from_jericho_env_batch
from lgat.q import EGreedy
from lgat.vocab import VocabDict


def main(args):
    game_name = os.path.split(args.game_dir)[-1]
    comment = f'_d-{game_name}_test-{args.test_game}'
    if args.name:
        comment += f'_{args.name}'
    writer = SummaryWriter(args.logdir, comment=comment)
    filepath = os.path.join(writer.log_dir, 'log.txt')
    file_handler = logging.FileHandler(filepath)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO,
                       format='[%(asctime)s] %(levelname)s - %(message)s',
                       handlers=handlers)
    logger = logging.getLogger(__name__)
    md_table = create_md_table_from_dic(vars(args), ['arg', 'value'])
    writer.add_text('args', md_table)
    logger.info(str(args))

    all_games = glob.glob(os.path.join(args.game_dir, r'*.z[0-9]'))
    test_game, train_games = None, []
    for g in all_games:
        if args.test_game in g:
            test_game = g
        else:
            train_games.append(g)

    logger.info(f"Train games: {train_games}")
    logger.info(f"Test game: {test_game}")
    env = JerichoMultiGameBatchEnv(train_games, max_episode_steps=args.max_steps)
    test_env = JerichoBatchEnv(test_game, batch_size=10, max_episode_steps=args.max_steps)
    test_env.env.name = "test_" + args.test_game

    encoder = RNNEncoder(tokenizer=args.tokenizer,
                         hidden_size=args.hidden_size,
                         max_length=args.tokenizer_max_length,
                         rnn=args.encoder)

    vocab = VocabDict(tokens=env.vocab,
                      max_token_length=env.max_word_length)

    if args.tdqn:
        tmpl_collection = TDQNTemplateCollection(train_games, args.tdqn_mask)
        generator = TemplateAction(encoder.size, tmpl_collection)
    else:
        generator = LGAT(encoder.size, vocab, args.masker)

    logger.info(f"Number of template: {generator.template_collection.template_num}")
    logger.info(f"Number of vocab: {generator.template_collection.vocab_size}")

    explorer = EGreedy(args.eps, args.eps_min, args.eps_decay)

    optimizer = torch.optim.Adam if True else torch.optim.RMSprop
    optimizer_params = {'lr': args.lr, 'weight_decay': args.lr_weight_decay}

    if args.replay_buffer == 'priority':
        memory = PrioritizedReplayBuffer(capacity=args.replay_capacity, error_max=2)
    else:
        memory = ReplayBuffer(capacity=args.replay_capacity)

    agent = Agent(
        encoder=encoder,
        act_generator=generator,
        explorer=explorer,
        memory=memory,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        exploration_bonus=args.exploration_bonus,
        reward_clip=(args.clip_min, args.clip_max),
        template_masking=(not args.tdqn and args.template_masking)
    )

    global_step = 0
    best_score = -float('inf')

    logger.info('Start training')
    for e in range(args.epoch):
        agent.train()

        obs, infos = env.reset()

        prev_obs, all_done, losses = obs, False, []
        finished, moves, rewards = np.zeros(env.batch_size), np.ones(env.batch_size), np.zeros(env.batch_size)
        pbar = tqdm.tqdm(total=args.max_steps,
                         desc=f'Epoch={e}', leave=False)

        while not all_done:
            prev_obs_hashes = env.world_state_hash
            masker_params, masker_params_list = create_masker_params_from_jericho_env_batch(infos)

            outputs, state = agent.forward(prev_obs, generate=True, prev_obs_hashes=prev_obs_hashes, masker_params=masker_params)

            action = outputs['action']
            action_str = outputs['generate']

            obs, reward, done, infos = env.step(action_str)

            for i in range(env.batch_size):
                if not finished[i]:
                    params = {
                        'action': action[i], 'state': prev_obs[i],
                        'next_state': obs[i], 'reward': reward[i],
                        'done': done[i], 'info': infos[i],
                        'action_str': action_str[i],
                        'masker_params': masker_params_list[i]
                    }

                    agent.remember(**params)

            if global_step % args.update_interval == 0:
                if len(agent.memory) >= args.replay_batch:
                    loss = agent.replay(args.replay_batch)
                    losses.append(loss)

            prev_obs = obs
            global_step += 1
            moves += 1 - np.array(done)
            all_done = all(done)
            finished = done
            pbar.update()
            agent.explorer.update()

        pbar.close()

        loss_avg = np.mean(losses) if losses else 0
        writer.add_scalar(f'loss', loss_avg, e)

        move_avg = np.mean(moves)
        writer.add_scalar('move', move_avg, e)

        score_avg, score_min, score_max = \
            env.write_score_to_tensorboard(writer, e)

        if args.explorer == 'boltzmann':
            writer.add_scalar('explorer/temperature',
                              explorer.temperature, e)
        else:
            writer.add_scalar('explorer/eps', explorer.eps, e)

        writer.add_scalar('lr', agent.scheduler.get_last_lr()[0], e)
        writer.add_scalar('replay_n', agent.replay_n, e)

        logger.info("Epoch:{}|Loss:{:.2f}|Move:{:.2f}|ScoreAvg:{:.2f}|"
                    "ScoreMin:{:d}|ScoreMax:{:d}|Explorer:{}"
                    .format(e, loss_avg, move_avg, score_avg, score_min,
                            score_max, agent.explorer))

        if not args.no_scheduler and len(agent.memory) >= args.replay_batch:
            agent.scheduler.step()

        agent.eval()
        with torch.no_grad():
            obs, infos = test_env.reset()

            prev_obs, all_done = obs, False
            moves, rewards = \
                np.ones(test_env.batch_size), np.zeros(test_env.batch_size)
            pbar = \
                tqdm.tqdm(total=args.max_steps,
                          desc=f'Epoch={e} Testing ...', leave=False)

            while not all_done:
                prev_obs_hashes = test_env.world_state_hash
                masker_params, masker_params_list = create_masker_params_from_jericho_env_batch(infos)

                outputs, state = agent.forward(prev_obs, generate=True, prev_obs_hashes=prev_obs_hashes, masker_params=masker_params)

                action_str = outputs['generate']

                obs, reward, done, infos = test_env.step(action_str)

                prev_obs = obs
                global_step += 1
                moves += 1 - np.array(done)
                all_done = all(done)
                pbar.update()

            pbar.close()

            best_env = test_env.max_score_env
            if best_env.score > best_score:
                agent.save(os.path.join(writer.log_dir, 'best_model.pt'))
                best_score = best_env.score

                with open(os.path.join(writer.log_dir, 'best_play.txt'), 'w') as f:
                    f.write(f'# Epoch={e}\n# Score={best_score}\n')
                    for i, h in enumerate(best_env.history):
                        f.write(f"# step {i}\n")
                        f.write(f">>> Obs:\n {h[0]}>>> Action: {h[1]}\n"
                                f">>> Score:{h[2]}\n\n")

            move_avg = np.mean(moves)
            writer.add_scalar('move/test', move_avg, e)

            score_avg, score_min, score_max = \
                test_env.write_score_to_tensorboard(writer, e, suffix=f'/{test_env.env.name}')

            logger.info("***Test score={}".format(score_avg))

    logger.info('Finish training')
    writer.close()
    env.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--test_game', default='zork1.z5', required=True)
    parser.add_argument('--game_dir', default=None, required=True)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--logall', action='store_true')
    parser.add_argument('--name', default=None)

    parser.add_argument('--tokenizer', default='bert-base-uncased', type=str)
    parser.add_argument('--tokenizer_max_length', default=256, type=int)
    parser.add_argument('--encoder', default='gru', choices=['lstm', 'gru'])
    parser.add_argument('--hidden_size', default=256, type=int)

    parser.add_argument('--vocab', default='game', type=str)
    parser.add_argument('--masker', nargs='+',
                        default=['pos', 'role', 'stopwords', 'observation_last', 'lm'], type=str)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_weight_decay', default=0, type=float)
    parser.add_argument('--no_scheduler', action='store_true')

    parser.add_argument('--explorer', default='egreedy',
                        choices=['boltzmann', 'egreedy'], type=str)
    parser.add_argument('--auto_decay_interval', default=1000, type=int)
    parser.add_argument('--eps', default=1.0, type=float)
    parser.add_argument('--eps_min', default=0.05, type=float)
    parser.add_argument('--eps_decay', default=0.99, type=float)

    parser.add_argument('--replay_buffer', default='priority', choices=['random', 'priority'])
    parser.add_argument('--replay_capacity', default=10000, type=int)
    parser.add_argument('--replay_batch', default=128, type=int)

    parser.add_argument('--clip_min', default=-10, type=int)
    parser.add_argument('--clip_max', default=10, type=int)

    parser.add_argument('--env_batch', default=10, type=int)
    parser.add_argument('--max_steps', default=100, type=int)

    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--update_interval', default=5, type=int)

    parser.add_argument('--playlog_interval', default=100, type=int)

    parser.add_argument('--exploration_bonus', action='store_true')

    parser.add_argument('--template_masking', action='store_true')

    parser.add_argument('--tdqn', action='store_true')
    parser.add_argument('--tdqn_mask', action='store_true')

    args = parser.parse_args()

    main(args)
