import time
import calendar

ticks = time.time()
print("当前时间戳: ", ticks)
#时间元组
localtime = time.localtime(time.time())
print("本地时间：", localtime)
#格式化时间
localtime = time.asctime(time.localtime(time.time()))
print("本地时间：", localtime)
#格式化日期
print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

cal = calendar.month(2018, 12)
print('以下输出2016年1月份的日历：')
print(cal)

print(calendar.isleap(2016))
