pip install imblearn
mv ./nohup.out ./nohup.out_$(date +%F-%H:%M)
nohup python3 ./train.py &
