# -*- coding: utf-8 -*-
"""
@author: jun_li
@contact: leoli19981021@163.com
@time: 2022/5/10 15:00
@description: Gurobi求解考虑顾客优先级的外卖配送车辆路径问题OPDPTWCPCS
"""

import gurobipy as gp
from gurobipy import GRB
from gurobipy import LinExpr
import pandas as pd
import math
import matplotlib.pyplot as plt

from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 10.5,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


"""数据读取"""
class Data:#数据类
    def __init__(self):
        self.businessNum = 0 # 商家数量
        self.customerNum = 0 # 客户数量
        self.nodeNum     = 0 # 点数量

        self.capacity    = 80 # 车辆容量
        self.TaskNo = []
        self.cor_X       = [] # x坐标
        self.cor_Y       = [] # y坐标
        self.demand      = [] # 客户需求
        self.service_time = [] # 服务时间
        self.early_time   = [] # 最早时间窗
        self.last_time     = [] # 最晚时间窗
        self.priority = [0] * self.nodeNum # 顾客优先级
        self.disMatrix   = [[]] # 距离矩阵


def read_txt_data(DataPath):
    # 读取车辆信息
    vehiclesInfo = pd.read_table(DataPath, nrows=1, names=['K','C','S'])
    # print(vehiclesInfo)
    # 读取Depot和任务信息
    columnNames = ['TaskNo','X','Y','Demand','ET','LT','ST','PI','DI']
    taskData = pd.read_table(DataPath, skiprows=[0], names = columnNames)

    # print(taskData['TaskNo'][75],taskData['X'][75])
    # print(taskData.iloc[0][0])
    """重建df 1-53商家 54-106顾客"""
    dict = {}
    for i in range(len(taskData)):
        # 如果为仓库
        if i == 0:
            dict[0] = [taskData['TaskNo'][i], taskData['X'][i], taskData['Y'][i], taskData['Demand'][i], taskData['ET'][i], taskData['LT'][i], taskData['ST'][i], taskData['PI'][i], taskData['DI'][i]]
        # 商家 1-53
        else:
            if taskData['DI'][i] > 0:
                dict[len(dict)] = [taskData['TaskNo'][i], taskData['X'][i], taskData['Y'][i], taskData['Demand'][i], taskData['ET'][i], taskData['LT'][i], taskData['ST'][i], taskData['PI'][i], taskData['DI'][i]]
    # 顾客 54-106
    for i in range(1, len(dict)):
        dict[len(dict)] = [taskData['TaskNo'][dict[i][-1]], taskData['X'][dict[i][-1]], taskData['Y'][dict[i][-1]], taskData['Demand'][dict[i][-1]], taskData['ET'][dict[i][-1]], taskData['LT'][dict[i][-1]], taskData['ST'][dict[i][-1]], taskData['PI'][dict[i][-1]], taskData['DI'][dict[i][-1]]]

    return dict


# 读取数据函数
def readData(data, path):
    dict = read_txt_data(path)
    dict[len(dict)] = dict[0]

    data.nodeNum = int(len(dict))  # 节点数量
    data.businessNum = int((data.nodeNum - 2) /2)
    data.customerNum = data.businessNum # 客户数量

    for i in range(len(dict)):
        data.cor_X.append(dict[i][1])
        data.cor_Y.append(dict[i][2])
        data.demand.append(dict[i][3])
        data.early_time.append(dict[i][4])
        data.last_time.append(dict[i][5])
        data.service_time.append(dict[i][6])
        data.priority.append(0)

    # 计算距离矩阵
    data.disMatrix = [([0] * data.nodeNum) for _ in range(data.nodeNum)]  # 初始化距离矩阵的维度,防止浅拷贝
    # data.disMatrix = [[0] * nodeNum] * nodeNum]; 这个是浅拷贝，容易重复
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            temp = (data.cor_X[i] - data.cor_X[j]) ** 2 + (data.cor_Y[i] - data.cor_Y[j]) ** 2
            data.disMatrix[i][j] = math.sqrt(temp)

    return data

def printData(data):
    print("下面打印数据\n")
    print("business number = %4d" % data.businessNum)
    print("customer number = %4d" % data.customerNum)
    print("vehicle capacity = %4d" % data.capacity)
    for i in range(len(data.demand)):
        print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(data.cor_X[i],data.cor_Y[i],data.demand[i], data.early_time[i], data.last_time[i], data.service_time[i], data.priority[i]))
    print("-------距离矩阵-------\n")
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            print("%6.2f" % (data.disMatrix[i][j]), end=" ")#保留2位小数
        print()


# reading data调用上面两个函数
data = Data()
path = r'LiLimPDPTWbenchmark/pdptw100_revised/lc101.txt'
readData(data, path)
printData(data)


"""建立模型"""
m = gp.Model('OPDPTW_TWO_INDEX')

"""添加变量"""
# binary--decision variable xijk
x = {}
# vehicle load
Q = {}
# the time that a vehicle starts servicing node i
B = {}
# the index of the first node in the route that visits node
v = {}
# 辅助变量
p = {}
w = {}
# 分段辅助变量0-1
y1 = {}
y2 = {}
y3 = {}
# 优先订单惩罚
u = {}
# 顾客满意度
g1 = {}
g2 = {}
ge = {}
gl = {}
# 定义xij
for i in range(data.nodeNum):
    for j in range(data.nodeNum):
        x[i,j] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x[%s,%s]' % (i,j))
# vehicle load
for i in range(data.nodeNum):
    Q[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Q[%s]' % i)
#
for i in range(data.nodeNum):
    B[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='B[%s]' % i)
#
for i in range(data.nodeNum):
    v[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='v[%s]' % i)

# 辅助变量
for i in range(data.nodeNum):
    p[i] = m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS, name='p[%s]' % i)

for i in range(data.nodeNum):
    w[i] = m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='w[%s]' % i)

# 分段辅助变量0-1
for i in range(data.nodeNum):
    y1[i] = m.addVar(lb=0,ub=1, vtype=GRB.BINARY, name='y1[%s]' % i)
for i in range(data.nodeNum):
    y2[i] = m.addVar(lb=0,ub=1, vtype=GRB.BINARY, name='y2[%s]' % i)
for i in range(data.nodeNum):
    y3[i] = m.addVar(lb=0,ub=1, vtype=GRB.BINARY, name='y3[%s]' % i)
# 优先订单惩罚
for i in range(data.nodeNum):
    u[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='u[%s]' % i)

# 顾客满意度
for i in range(data.nodeNum):
    g1[i] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='g1[%s]' % i)
for i in range(data.nodeNum):
    g2[i] = m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='g2[%s]' % i)
for i in range(data.nodeNum):
    ge[i] = m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ge[%s]' % i)
for i in range(data.nodeNum):
    gl[i] = m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ge[%s]' % i)


"""更新模型"""
m.update()


"""设置目标函数"""
# 1 cij * xij
'''目标函数1：车辆行驶成本'''
obj1 = LinExpr(0)
for i in range(data.nodeNum):
    for j in range(data.nodeNum):
        if(i != j):
            obj1.addTerms(data.disMatrix[i][j], x[i,j])

m.setObjectiveN(obj1, index=0, weight=1,  name='obj1')
# 2 x0j
'''目标函数2：车辆启动固定成本'''
obj2 = LinExpr(0)  # 线性表达式构造函数
for j in range(1, data.nodeNum):  # j属于C
    obj2.addTerms(50, x[0, j])  # 向线性表达式中添加新项。启动成本50

m.setObjectiveN(obj2, index=1, weight=1, name='obj2')

# 3 人力成本 tij
'''目标函数3：人力成本'''
obj3 = LinExpr(0)
for i in range(data.nodeNum):
    for j in range(data.nodeNum):
        if(i != j):
            obj3.addTerms(data.service_time[j], x[i,j])
obj3 = obj1 + obj3

m.setObjectiveN(obj3, index=2, weight=1, name='obj3')
#
'''目标函数4：优先订单'''
obj4 = LinExpr(0)
for i in range(data.nodeNum):
    if data.priority[i] == 1:
        """改为分段"""
        obj4.addTerms(1,u[i])

m.setObjectiveN(obj4, index=3, weight=1, name='obj4')

'''目标函数5：顾客满意度'''
# obj5 = LinExpr(0)
# for i in range(data.nodeNum):
#     obj5.addTerms(3,ge[i])
#
# m.setObjectiveN(obj5, index=4, weight=1, name='obj5')

obj5 = LinExpr(0)
for i in range(data.nodeNum):
    obj5.addTerms(5,gl[i])

m.setObjectiveN(obj5, index=4, weight=1, name='obj5')


"""添加约束条件"""
P = range(1,data.businessNum+1)
D = range(data.businessNum+1,data.nodeNum-1)
PUD = range(1,data.nodeNum-1)
N = range(data.nodeNum)


# 2 可以
for j in PUD:
    st = LinExpr(0)
    for i in N:
        st.addTerms(1,x[i,j])
    m.addConstr(st==1, name='st1')


# 3 可以
for i in PUD:
    st = LinExpr(0)
    # for j in N:
    for j in range(0,data.nodeNum-1):
        st.addTerms(1,x[i,j])
    # m.addConstr(st==1, name='st2')
    m.addConstr(st<=1, name='st2')


# 4 加上服务时间
M = 100000
for i in N:
    for j in N:
        m.addConstr(B[j] >= B[i] + data.disMatrix[i][j] - M * (1 - x[i,j]) + data.service_time[i],name='st3')


# 5
for i in N:
    for j in N:
        m.addConstr(Q[j] >= Q[i] + data.demand[j] - M * (1 - x[i,j]),name='st4')


# 7
demand = [0, 10, 10, 20, 20, 10, 10, 30, 40, 20, 10, 10, 10, 40, 20, 10, 10, 30, 40, 10, 10, 30, 20, 10, 10, 10, 10, 10, 10, 20, 40, 30, 40, 20, 50, 10, 10, 10, 10, 20, 10, 20, 10, 30, 20, 20, 10, 20, 10, 20, 10, 30, 20, 20, -10, -10, -20, -20, -10, -10, -30, -40, -20, -10, -10, -10, -40, -20, -10, -10, -30, -40, -10, -10, -30, -20, -10, -10, -10, -10, -10, -10, -20, -40, -30, -40, -20, -50, -10, -10, -10, -10, -20, -10, -20, -10, -30, -20, -20, -10, -20, -10, -20, -10, -30, -20, -20, 0]

for i in N:
    m.addConstr(max(0, demand[i]) <= Q[i], name='R7')
    m.addConstr(Q[i] <= min(data.capacity,data.capacity+demand[i]), name='R8')


# 8 加上服务时间
for i in P:
    m.addConstr(B[data.businessNum+i] >= B[i] + data.disMatrix[i][data.businessNum+i] + data.service_time[i])


# 9
for i in P:
    m.addConstr(v[data.businessNum+i] == v[i])


# 10
for j in PUD:
    m.addConstr(v[j] >= j * x[0,j])


# 11
for j in PUD:
    m.addConstr(v[j] <= j * x[0,j] - data.businessNum * (x[0,j] - 1))


# 12
for i in PUD:
    for j in PUD:
        m.addConstr(v[j] >= v[i] + data.businessNum * (x[i,j] - 1))


# 13
for i in PUD:
    for j in PUD:
        m.addConstr(v[j] <= v[i] + data.businessNum * (1 - x[i,j]))


# 14 不回2n+1点
lhs = LinExpr(0)
for j in PUD:
    lhs.addTerms(1, x[j, data.nodeNum - 1])
m.addConstr(lhs == 0)


# 15 确定有向图的方向，车辆从0点出发，而不是2n+1点
lhs = LinExpr(0)
for j in PUD:
    lhs.addTerms(1, x[data.nodeNum - 1,j])
m.addConstr(lhs == 0)


for i in N:
    m.addConstr(p[i] == B[i] - data.last_time[i])


for i in N:
    m.addGenConstrMax(w[i],[0,p[i]])


for i in N:
    m.addConstr(u[i] == 3*y1[i]+6*y2[i]+12*y3[i])


# 线性辅助变量
"""等于改为小于等于"""
for i in N:
    m.addConstr(y1[i]+y2[i]+y3[i] <= 1)


for i in N:
    m.addConstr(w[i] <= 10*y1[i]+20*y2[i]+M*y3[i])


for i in N:
    m.addConstr(w[i] >= 10*y2[i]+20*y3[i]-10E-9)


for i in N:
    # 0 < g2 <= 1
    m.addConstr(g2[i] == (1 - (data.last_time[i] * 1.25 - B[i]) / (data.last_time[i] * 1.25 - data.last_time[i] + 10e-9)))


for i in N:
    m.addGenConstrMax(gl[i],[0,g2[i]])


m.presolve()
m.Params.TimeLimit = 100
m.setParam("Heu*", 0.5)
m.optimize()

"""可视化"""

'''输出多目标中各目标值'''
# Read and solve a model with multiple objectives
# get the set of variables
xx = m.getVars()
# Ensure status is optimal
# assert m.Status == GRB.Status .OPTIMAL
# Query number of multiple objectives , and number of solutions
nSolutions = m.SolCount
nObjectives = m.NumObj
print("-----优化目标和可行解的数量-----")
print ('Problem has', nObjectives , 'objectives' )
print ('Gurobi found', nSolutions , 'solutions' )
print("-----各优化目标值-----")
# For each solution , print value of first three variables , and
# value for each objective function
solutions = []
for s in range (nSolutions):
    # Set which solution we will query from now on
    m.params.SolutionNumber = s
    # Print objective value of this solution in each objective
    print ('可行解', s+1 , ':', end =' ')
    yobj = 0
    for o in range (nObjectives):
        # Set which objective we will query
        m.params.ObjNumber = o
        yobj += m.ObjNVal
        # Query the o-th objective value
        print('第%d个目标值为'%(o+1),m.ObjNVal, end =' ')

    print('\n######################################')
    print('多目标函数值%d为：'%(s+1), yobj, end='\n######################################')
    print('\n')


for key in x.keys():
    if(x[key].x > 0 ):
        print(x[key].VarName + ' = ', x[key].x)


"""结果处理"""
'''字符转为list'''
zifu = []
for key in x.keys():
    if (x[key].x > 0):
        zifu.append(x[key].VarName)
zifu.sort()
print(zifu)

'''提取数字'''
import re
fenge = [([]) for i in range(data.nodeNum-2)]
for char in range(data.nodeNum-2):
    a = zifu[char]
    number = list(filter(str.isdigit, a))
    fenge[char] = re.findall(r"\d+\.?\d*", a)
print('提取数字',fenge)


import copy
routes = []

num = copy.deepcopy(fenge)
for fen in fenge:
    if fen[0] == '0':
        routes.append(fen)
        num.remove(fen)


check = []
for a in range(len(num)):
    check.append(num[a][0])


for route in routes:
    i = 1
    j = 0
    n = len(num)
    while j <= n:
        if route[i] == num[j][0]:
            route.append(num[j][1])
            # num.remove(num[j])
            i += 1
            j = 0
        else:
            j += 1
        if route[i] not in check:
            break
    print(route)


routes[0] = [int(x) for x in routes[0]]
routes[1] = [int(x) for x in routes[1]]
routes[2] = [int(x) for x in routes[2]]
routes[3] = [int(x) for x in routes[3]]
routes[4] = [int(x) for x in routes[4]]
routes[5] = [int(x) for x in routes[5]]
routes[6] = [int(x) for x in routes[6]]
routes[7] = [int(x) for x in routes[7]]
routes[8] = [int(x) for x in routes[8]]
routes[9] = [int(x) for x in routes[9]]


'''画图'''
plt.figure(dpi=300)


for i in range(len(routes)):
    x_coor_list, y_coor_list = [], []
    for j in routes[i]:
        x_coor_list.append(data.cor_X[j])
        y_coor_list.append(data.cor_Y[j])
        if data.nodeNum > j > data.businessNum:
            mark = 'o'
        else:
            mark = 's'

        if 0 < j < data.nodeNum:
            plt.scatter(data.cor_X[j],data.cor_Y[j],color='tab:blue', marker=mark,s=10)

    plt.plot(x_coor_list,y_coor_list,linewidth=0.8,color='tab:blue')
#

plt.scatter(data.cor_X[0],data.cor_Y[0],marker='*',color='red',s=200,label='配送中心')
plt.xticks(fontproperties='Times New Roman')
plt.yticks(fontproperties='Times New Roman')
plt.xlabel('x_coord', fontproperties='Times New Roman')
plt.ylabel('y_coord', fontproperties='Times New Roman')
plt.legend()







