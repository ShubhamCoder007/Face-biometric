f = open("t.txt")
a = f.read()
a = a[1:-1]
a=a.split(sep=',')
dic = {}
for i in a:
	i = i.replace(" ","")
	print(i)
	
	i = i.split(sep=':')
	i[0]=i[0].replace("'","")
	dic[int(i[1])]=i[0]
	
