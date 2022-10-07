import re
f = open("/home/lzy/VDG/SpCL/logs/base_0.9reliability_market_update_real_momentum0.4/log.txt")
lines = f.readlines()
for line in lines:
    # map= re.search(r'top-1 \d\.]*?)', line)
    # map = re.match('\Stop-1     *', line)
    if "top-1" in line:
        print(line[-6:-2])
    if "Mean AP" in line:
        print(line[-6:-2], "map")