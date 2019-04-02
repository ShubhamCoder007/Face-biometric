import os

roll = input('Enter the roll:')
os.mkdir('dataset/' + roll)
os.mkdir('dataset/' + roll + '/training_set')
os.mkdir('dataset/' + roll + '/test_set')

