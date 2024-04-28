function str=cell2str(cel)
% 输入cell类型矩阵表格，返回str类型矩阵表格
% 将cell数据类型转化为str类型
% 想法：用元组读取xlsx写入文本,读取文本格式是string,再用split根据分隔符进行分割string

% 把a写入ing.txt文本中
writecell(cel,'ing.txt','Delimiter','|')
% 分隔符必须为以下字符之一: ' '、'\t'、','、';'、'|'，或者与它们对应的字符名称:
% 'space'、'tab'、'comma'、'semi' 或 'bar'。

% a读取文本内容格式为string(n行1列)
a=readlines("ing.txt");

% 用split把a按照分隔符分割存入str中
% 为什么是a(1:end-1),原因是存入文本是多了一个空行
str=split(a(1:end-1),'|');

% 删除中间文本
delete("ing.txt")