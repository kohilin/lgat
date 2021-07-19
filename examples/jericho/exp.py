import json
import pandas as pd
import sys

from jericho.defines import ABBRV_DICT


sys.path.append('../../')
from lgat.action.masker import create_masker_params_from_jericho_env_batch


MOVE_ACTIONS = ['n', 's', 'w', 'e', 'ne', 'nw', 'se', 'sw', 'u', 'd', 'north',
                'south', 'west', 'east', 'northwest', 'northeast', 'southwest',
                'southeast', 'up', 'down']


def create_md_table_from_count_dict(d, max_row=100, name='action'):
    md_table = f'|{name}|count| \n|:---|:---| \n'

    for idx, (k, v) in enumerate(sorted(d.items(),
                                        key=lambda x: x[1], reverse=True)):
        if idx == max_row:
            break
        md_table += f'|{k}|{v}| \n'

    return md_table


def create_md_table_from_dic(d, headers, max_rows=None):
    md_table = '|'
    for h in headers:
        md_table += h + '|'
    md_table += ' \n'
    for i in range(len(headers)):
        md_table += '|:---'
    md_table += '|\n'

    for idx, (k, v) in enumerate(d.items()):
        if idx == max_rows:
            break
        md_table += f'|{k}|{v}| \n'

    return md_table


def walkthrough_exp(args, env, tmpl_collection):
    obs, infos = env.reset()

    tar_steps = args.test_walkthrough_target_steps
    success_n, failed_steps, statues = 0, [], []
    walkthrough_n = 0

    for idx, a in enumerate(env.walkthrough):
        if tar_steps and idx not in tar_steps:
            obs, reward, done, infos = env.step([a] * args.env_batch)
            continue

        if not env.env.is_change_world_state_hash(a) and args.test_walkthrough_ignore:
            print('Ignore an action: ', a)
            obs, reward, done, infos = env.step([a] * args.env_batch)
            continue

        walkthrough_n += 1

        print('Step', idx)

        _, masker_params_list = create_masker_params_from_jericho_env_batch(infos)
        masker_params = masker_params_list[0]

        a_norm = a.lower()

        if a_norm in MOVE_ACTIONS:
            a_norm = 'go ' + ABBRV_DICT.get(a_norm, a_norm)

        tmp = []
        for w in a_norm.split():
            tmp.append(ABBRV_DICT.get(w, w))
        a_norm = ' '.join(tmp)

        is_success, status = \
            tmpl_collection.is_generable(a_norm, **masker_params)

        if args.test_walkthrough_retry and not is_success:
            found_same_transition = False

            is_success, status = tmpl_collection.is_generable('get get get get', # use all slots
                                                              verbose=False,
                                                              **masker_params)

            for tmpl, target_tmpl_masks in status.items():
                target_tmpl_masks = status[tmpl]
                for v in target_tmpl_masks['v']['tokens']:
                    for o1 in target_tmpl_masks['o1']['tokens']:
                        for p in target_tmpl_masks['p']['tokens']:
                            for o2 in target_tmpl_masks['o2']['tokens']:
                                c = ' '.join([v, o1, p, o2])
                                c = c.replace('Null', '')
                                is_same = env.envs[0].be_same_transition(a, c)
                                if is_same:
                                    print(f'Found the same transition with [{c}] for [{a}]')
                                    found_same_transition = True
                                    status[tmpl]['success'] = True
                                    break
                            if found_same_transition: break
                        if found_same_transition: break
                    if found_same_transition: break
                if found_same_transition: break

            if found_same_transition:
                is_success = True

        if is_success:
            success_n += 1
        else:
            failed_steps.append((idx, a))
        statues.append((a, is_success, status))

        if not is_success and args.test_walkthrough_debug:
            while True:
                print('>>> action (type "next" to proceed): ', end='')
                x = input()
                if x == 'next':
                    break
                elif x == 'obs':
                    print(infos[0]['state_description'])
                    continue
                else:
                    if x.startswith('!'):
                        tmpl = x.replace('!', '').split('|')

                        t = tmpl_collection.get(tmpl[0])
                        if not t:
                            print(f'Unknown tmplate name: {tmpl}')
                            print(f'Availables: {tmpl_collection.templates}')
                            continue

                        if len(tmpl) == 1 and t:
                            verb = t.v.words[0]
                        else:
                            verb = tmpl[1]
                        tmpl = tmpl[0]

                        x = f'{verb} Null Null Null'
                        is_success, status = \
                            tmpl_collection.is_generable(x, verbose=False,
                                                         **masker_params)

                        if tmpl not in status:
                            print('No verb found')
                            continue

                        target_tmpl_masks = status[tmpl]
                        possible_commands = []
                        for v in target_tmpl_masks['v']['tokens']:
                            for o1 in target_tmpl_masks['o1']['tokens']:
                                for p in target_tmpl_masks['p']['tokens']:
                                    for o2 in target_tmpl_masks['o2']['tokens']:
                                        possible_commands.append(
                                            ' '.join([v, o1, p, o2]))
                                possible_commands.append(' '.join([v, o1]))

                        possible_commands = set(possible_commands)

                        print(f'Possible commands: ', possible_commands)

                        found_same_transition = False
                        for c in possible_commands:
                            c = c.replace('Null', '')
                            is_same = env.envs[0].be_same_transition(a, c)
                            if is_same:
                                print(f'✅ Found the same transition with [{c}].')
                                found_same_transition = True
                                break

                        if not found_same_transition:
                            print(f'❌ Not found the same transition.')

                    else:
                        is_success, status = \
                            tmpl_collection.is_generable(x, **masker_params)

                        x = x.replace('Null', '')

                        is_same = env.envs[0].be_same_transition(a, x)
                        print('be_same_transition -> ', is_same)

        obs, reward, done, infos = env.step([a] * args.env_batch)

    if not args.test_walkthrough_debug:
        txt_name = args.game + '.test_walkthrough.txt'
        if args.name:
            txt_name += '.' + args.name

        with open(txt_name, 'w') as f:
            f.write(f'Walkthrough num: {walkthrough_n}\n')
            f.write(f'Success num: {success_n}\n')
            f.write(f'Failed num: {walkthrough_n - success_n}\n')
            r = 0 if not walkthrough_n else success_n / walkthrough_n
            f.write(f'Success ratio: {r}\n', )
            if len(failed_steps):
                f.write(
                    f'Failed steps:\n{" ".join([str(f[0]) for f in failed_steps])}\n')
                for fs in failed_steps:
                    f.write(f'{fs}\n')

        lines = []
        for idx, s in enumerate(statues):
            action, is_success, status = s
            if is_success:
                tmpl, results = \
                [(k, v) for (k, v) in status.items() if v['success']][0]
                line = [
                           args.game,
                           action,
                           is_success,
                           tmpl,
                       ] + [results[k]['n'] for k in ['v', 'o1', 'p', 'o2']]
                lines.append(line)
            else:
                line = [args.game, action, is_success, None, None, None, None, None]
                lines.append(line)

        df = pd.DataFrame(lines,
                          columns=['game', 'action', 'is_success', 'template', 'v', 'n1', 'p', 'n2'])
        csv_name = args.game + '.test_walkthrough.csv'
        if args.name:
            csv_name += '.' + args.name
        df.to_csv(csv_name, index=False)

        with open(csv_name + '.status_json', 'w') as f:
            json.dump(statues, f)
