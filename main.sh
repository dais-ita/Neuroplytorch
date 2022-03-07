for E in basic_neuro_experiment basic_neuro_experiment_emnist basic_neuro_experiment_slow basic_neuro_experiment_emnist_slow
do 
    python main.py --name $E
done