s1 = '())))))))((())(()'
count_left=0
count_right = 0
count_all = 0
for i in range(len(s1)):
    if s1[i]=='(':
        if count_right!=0 and count_left!=0:
            count_all+= count_left+count_right-1
            count_right=0
            count_left=0
        else:
            count_left+=1
    else:
        if count_left>0:
            count_right+=1
        else:continue
if count_right!=0 and count_left!=0:
    count_all+= count_left+count_right-1
print(len(s1)-count_all)