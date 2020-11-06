import run_ludo

for i in ["reverse","same","monte"]:
    print("-----------------{}---------------------".format(i))
    run_ludo.train(50001,i)