ps -ef |grep "python3 app.py"|grep -v grep |awk '{print $2}'|xargs kill -9
rm nohup.out
nohup python3 dev_test.py &
