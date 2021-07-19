- Download [Jericho games](https://github.com/microsoft/jericho) and put `*.z
  [0-9]` files in `games/` (you can use any name)
- Install python 3.7 or higher
  
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('framenet_v17'); nltk.download('verbnet')"
python -m spacy download en
```

- Commands
```bash
# L-GAT
python run.py --game games/zork1.z5

# TDQN+ (re-implementation of TDQN, https://github.com/microsoft/tdqn)
python run.py --game games/zork1.z5 --tdqn

# Train a single L-GAT agent with all games in games/
python run.py --game_dir games

# Train a single TDQN+ agent with all games in games/
python run.py --game_dir games --tdqn

# Train a single L-GAT agent withh agmes in games/ except for zork1.z5
python run_cross.py --test_game zork1.z5 --game_dir ./games

# Train a single TDQN+ agent with agmes in games/ except for zork1.z5
python run_cross.py --test_game zork1.z5 --game_dir ./games --tdqn

# tensorboard
tensorboard --logdir ./runs

# evaluate
python evaluate.py --tb_event_dir ./runs --plot_save_dir ./plot

# walkthrough
python run.py --game games/zork1.z5 --test_walkthrough --test_walkthrough_retry
```

