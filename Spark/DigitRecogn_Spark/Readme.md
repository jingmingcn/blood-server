
#����Spark��Ocr��д�ַ�ʶ��ϵͳDemo

##����ѵ����������
###�������ݼ�

wget http://labfile.oss.aliyuncs.com/courses/593/data.csv
```
�����ݼ���https://www.shiyanlou.com/courses/593/labs/1966/document ��Ϊ����������ѵ���õ����ݼ�
###��ʽ�����ݼ�
Spark���ѧϰ���õ�����ѵ�����ݸ�ʽΪLabeled point��LibSVM���ڴˣ�����ʹ��Labeled point��Ϊѵ�����ݸ�ʽ��


labeled point ��һ���ֲ�������Ҫô���ܼ��͵�Ҫô��ϡ���͵ģ���һ��label/response���й�������Spark�labeled points �������ලѧϰ�㷨������ʹ��һ��double�����洢һ��label����������ܹ�ʹ��labeled points���лع�ͷ��ࡣ
�ڶ����Ʒ����һ��label������ 0������������ 1�����������ڶ༶�����У�labels������class����������0��ʼ��0,1,2,......

��Demo�������ر�Ҷ˹��Ϊѵ����Ԥ��ģ�ͣ�����ֵ�����ǷǸ�����

���������й������ȶ�ȡ����ʽ��./data.csv�е����ݣ�Ȼ�����ҳǰ�˴�����ѵ������һ���ʽ��Ϊlabeled points��ʽ
�����ɵ�LabeledPoints���ݱ�����LabeledPointsdata.txt�С�

��ҪԤ��ʱ���Ƚ�LabeledPointsdata.txt�е����ݶ�ȡΪSpark ר�� RDD ��ʽ��Ȼ��ѵ����model��

##����


###����������
```
 python -m SimpleHTTPServer 3000
```

###���ط�����
```
 python server.py

```
###����
```
 localhost:3000
```