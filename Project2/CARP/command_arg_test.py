import getopt
import sys

argv = sys.argv[1:]
username = ''
password = ''

try:
    opts, _ = getopt.getopt(argv, 'hu:p:', ['help', 'username=', 'password='])
except getopt.GetoptError:
    print('Error: test_arg.py -u <username> -p <password>')
    print('   or: test_arg.py --username=<username> --password=<password>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print('command_arg_test.py -u <username> -p <password>')
        print('or: command_arg_test.py --username=<username> --password=<password>')
        sys.exit()
    elif opt in ("-u", "--username"):
        username = arg
    elif opt in ("-p", "--password"):
        password = arg
print('username为：', username)
print('password为：', password)
