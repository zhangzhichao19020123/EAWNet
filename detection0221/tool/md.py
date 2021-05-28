from hashlib import md5
filename =  'P0003__1__0___0.jpg'
m = md5(filename.encode("gb2312"))
#a=int (m.hexdigest(),16)
a=int(m.hexdigest(),16) & ((1<<63)-1) 
print (a)