import evaluate
print(evaluate.load("accuracy").compute(predictions=[0,1,1], references=[0,1,0]))

