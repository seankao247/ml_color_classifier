# ml_color_classifier

samplecode_train.c
=>
/*
研究參考網站code的訓練程式
https://www.the-diy-life.com/running-an-artificial-neural-network-on-an-arduino-uno/
這個程式共只能三層　hidden 只有一層 加上 input output 共三層
*/

徐_main.c
=>
徐睿桐之前做 加速度ML的training code
加入我註解的原始版本
相當於在改之前做個備份

徐_main_v2.c
=>
徐_main.c的修改版本，加入我的修改，把它變得可以跑我們的資料
成功可跑

資料格式:
float train_data_input[train_data_num][InputNodes_num] = {
    {479, 468, 261},  //(R,G,B) of pattern 1
    {486, 482, 266},  //(R,G,B) of pattern 2
    {488, 479, 265},
    {490, 477, 267},
    {480, 471, 262},
    {482, 472, 264}}

int train_data_output[train_data_num][target_num] = {
    {0, 0, 0, 1},  
    {0, 0, 0, 1},
    {0, 0, 0, 1},
    {0, 0, 0, 1},
    {0, 0, 0, 1},
    {0, 0, 0, 1}}
