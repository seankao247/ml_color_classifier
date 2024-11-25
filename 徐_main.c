/*
這個經過自己註解後的 main_previous.c
相當於有我註解的原始版本
*/
//Author: Ralph Heymsfeld
//28/06/2018

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
// #include <NUC100Series.h>

/******************************************************************
 * dataset 格式設定
 ******************************************************************/

#define max_data_len 400 //每筆raw data的最大長度
#define new_length 50 //經處理後data的固定長度
#define train_data_num 40 //train data總數 : 隨著收更多資料需要更新這個數值
#define test_data_num 20
#define target_num 10 //output可能答案總數

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/
#define HiddenNodes 64

const float LearningRate = 0.1;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float goal_acc = 0.95; //期望的準確度

// 創建train dataset,並設定offset=0
float train_data_input[train_data_num][new_length * 3];
int train_data_output[train_data_num][target_num];

//測試用test dataset
float test_data_input[test_data_num][new_length * 3];
int test_data_output[test_data_num][target_num];

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int ReportEvery1000;
int RandomizedIndex[train_data_num];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float Hidden[HiddenNodes];
float Output[target_num];
float HiddenWeights[new_length * 3+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][target_num];
float HiddenDelta[HiddenNodes];
float OutputDelta[target_num];
float ChangeHiddenWeights[new_length * 3+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][target_num];

int target_value;
int out_value;
int max;



void normalize(float *data, int length) // 將數列正規化到0到1之間
{
    int i;
    float min = data[0];
    float max = data[0];
    // 找出數列中的最小值和最大值
    for (int i = 1; i < length; i++) {
        if (data[i] < min) {
            min = data[i];
        }
        if (data[i] > max) {
            max = data[i];
        }
    }

    for (int i = 0; i < length; i++) {
        data[i] = (data[i] - min) / (max - min);
    }
} 

void scale_data(float (*data), int length) {
    int i;
    float factor = (float)(length - 1) / (new_length - 1);
    float *new_data;
    float max, min, mean;
    if((new_data=malloc(new_length * sizeof(float)))==NULL)
        printf("malloc error\n");
    for (i = 0; i < new_length; i++) {
        float index = i * factor;
        int lower_index = (int)index;
        int upper_index = lower_index + 1;
        float weight = index - lower_index;
        if (upper_index >= length) {
            new_data[i] = (data)[lower_index];
        } else {
            new_data[i] = (1 - weight) * (data)[lower_index] + weight * (data)[upper_index];
        }
    }
    // data正規化
    normalize(new_data,new_length);
    memcpy(data, new_data, new_length * sizeof(float));
    free(new_data);
}

int train_preprocess() //要改variable名稱
{
    int i, j;
    int success_flag = 0;
    FILE *fp;
    int count;
    int x_no=0,y_no,z_no,c;
    char filename[65]; //配給檔案路徑名稱記憶體，array大小隨你路徑長度調整
    char data_buf[20]; //從txt接入data的buffer

    float datax[max_data_len],datay[max_data_len],dataz[max_data_len];

    memset(train_data_output, 0, sizeof(train_data_output));//把train_data_output給定offset=0
    for (i = 1; i <= train_data_num;i++)
    {
        sprintf(filename, "data_train//DATA%d.txt", i); //設定檔案路徑名稱
        fp = fopen(filename,"r");
        if(fp==NULL)
        {
            printf("open file fail\n");
            break;
        }
        printf("DATA%d 讀取中\n",i);
        count = 0;
        while(fgets(data_buf, 20, fp) != NULL)
        {
            if (count >= max_data_len) //判斷數據是否超過最大個數
            {
                printf("數據過多，無法全部讀取！\n");
                break;
            }
            c = 0;
            for(j=0;data_buf[j];j++)
            {
                if(data_buf[j]==' ')
                {
                    data_buf[j]='\0';
                    if(c==0)
                        y_no=j+1;
                    else
                        z_no=j+1;
                    c++;
                }                
            }

            datax[count] = atof(data_buf+x_no);//將字符串轉換為浮點數並存儲到陣列中
            datay[count] = atof(data_buf+y_no); 
            dataz[count] = atof(data_buf+z_no);
            count++; 
        }
        fclose(fp);

        //資料長度一致化;
        scale_data(datax, count);;
        scale_data(datay, count);
        scale_data(dataz, count);

        //設定train data input
        for (int j = 0; j < 50; j++) 
        {
            train_data_input[i-1][j] = datax[j];
            train_data_input[i-1][j + 50] = datay[j];
            train_data_input[i-1][j + 100] = dataz[j];
        }

        //標記train data output
        if(i%10==0) 
        {
            train_data_output[i-1][9]=1;
        }
        else
        {
            train_data_output[i-1][i%10-1]=1;
        }
        printf("DATA%d 資料寫入dataset成功\n", i);
        if(i==20)
            success_flag = 1;
    }
    if(success_flag)
    {
        printf("training dataset建立成功\n");
        return 0;
    }
    else
    {
        printf("training dataset建立失敗\n");
        return 1;
    }
}

int test_preprocess()
{
    int i, j;
    int success_flag = 0;
    FILE *fp;
    int count;
    int x_no=0,y_no,z_no,c;
    char filename[65];
    char data_buf[20]; 
    float datax[max_data_len],datay[max_data_len],dataz[max_data_len];
    memset(test_data_output, 0, sizeof(test_data_output));
    for (i = 1; i <= test_data_num;i++)
    {
        sprintf(filename, "data_test//DATA%d.txt", i); //設定檔案路徑名稱
        fp = fopen(filename,"r");
        if(fp==NULL)
        {
            printf("open file fail\n");
            break;
        }
        printf("DATA%d 讀取中\n",i);
        count = 0;
        while(fgets(data_buf, 20, fp) != NULL)
        {
            if (count >= max_data_len)
            {
                printf("數據過多，無法全部讀取！\n");
                break;
            }
            c = 0;
            for(j=0;data_buf[j];j++)
            {
                if(data_buf[j]==' ')
                {
                    data_buf[j]='\0';
                    if(c==0)
                        y_no=j+1;
                    else
                        z_no=j+1;
                    c++;
                }                
            }
            datax[count] = atof(data_buf+x_no);
            datay[count] = atof(data_buf+y_no); 
            dataz[count] = atof(data_buf+z_no);
            count++; 
        }
        fclose(fp);
        scale_data(datax, count);;
        scale_data(datay, count);
        scale_data(dataz, count);
        for (int j = 0; j < 50; j++) 
        {
            test_data_input[i-1][j] = datax[j];
            test_data_input[i-1][j + 50] = datay[j];
            test_data_input[i-1][j + 100] = dataz[j];
        }
        if(i%10==0) 
        {
            test_data_output[i-1][9]=1;
        }
        else
        {
            test_data_output[i-1][i%10-1]=1;
        }
        printf("DATA%d 資料寫入test dataset成功\n", i);
        if(i==20)
            success_flag = 1;
    }

    if(success_flag)
    {
        printf("testing dataset建立成功\n");
        return 0;
    }
    else
    {
        printf("testing dataset建立失敗\n");
        return 1;
    }
}

void run_train_data()   //這個程式改至原本的toTerminal()，用途是預測並印出 只會用在最後一次 (by sean) 
{
    int i, j, p, q, r;
    int correct=0;
    //float accuracy = 0; //為了方便閱讀 我把這行移到最下面 應該沒事(by sean)
    printf("Train result:\n");
    for( p = 0 ; p < train_data_num ; p++ )
    {   
        //這邊應該是找出label，即target_value (by sean)
        max = 0;
        for (int i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[new_length * 3][i] ;
            for( j = 0 ; j < new_length * 3 ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        // 找出最大值 (我猜測意思是找機率最大的那個作為分類答案(by sean))
        max = 0;
        for (int i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        float accuracy = 0;//上面移到這(by sean)
        accuracy = (float)correct / train_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);

}

void run_test_data()    //應該是跟run_train_data()一模一樣的功能，只是data換成test所以多寫一個 只會用在最後一次(by sean)
{
    int i, j, p, q, r;
    int correct=0;
    float accuracy = 0;
    printf("Test result:\n");
    for( p = 0 ; p < test_data_num ; p++ )
    { 
        max = 0;
        for (int i = 1; i < target_num; i++) 
        {
            if (test_data_output[p][i] > test_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[new_length * 3][i] ;
            for( j = 0 ; j < new_length * 3 ; j++ ) {
                Accum += test_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        // 找出最大值
        max = 0;
        for (int i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);    //這邊文字應該改成test pattern?
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / test_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

int setup(){
    int i, j, p,ret;
    //UART0_Init();
    srand(time(NULL)); //當前時間的值作為生成器的種子, 可改!
    ReportEvery1000 = 1;
    for( p = 0 ; p < train_data_num ; p++ ) 
    {    
        RandomizedIndex[p] = p ;
    }
    ret=train_preprocess();
    ret|=test_preprocess();
    if(ret) //有出錯
        return 1;
    //print dataset
    /*
    for (i = 0; i < test_data_num; i++)
    {
        printf("\nDATA[%d] input:\n",i+1);
        for (j = 0; j < 3*new_length;j++)
            printf("%.2f ", test_data_input[i][j]);
        printf("\nDATA[%d] output:\n",i+1);
        for (j = 0; j < target_num;j++)
            printf("%d ", test_data_output[i][j]);
    }
    */
    printf("\ntrain DATA[%d] output:\n",40);
    for (j = 0; j < target_num;j++)
        printf("%d ", train_data_output[39][j]);
    printf("\ntest DATA[%d] output:\n",20);
    for (j = 0; j < target_num;j++)
        printf("%d ", test_data_output[19][j]);
    return 0;
}  

float Get_Train_Accuracy()  //跟run_train_data 一模一樣的code 順序不一樣而已 超奇怪
                            //差別只是run_train_data()是印出，Get_Train_Accuracy()是return    (by sean)
{
    int i, j, p, q, r;
    int correct = 0;
    for (p = 0; p < train_data_num; p++)
    {
/******************************************************************
* Compute hidden layer activations
******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[new_length * 3][i] ;
            for( j = 0 ; j < new_length * 3 ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

/******************************************************************
* Compute output layer activations
******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        //get target value
        max = 0;
        for (int i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        //get output value
        max = 0;
        for (int i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;
        //compare output and target
        if (out_value==target_value)
        {
            correct++;
        }
    }

    // Calculate accuracy
    float accuracy = (float)correct / train_data_num;
    return accuracy;
}

//待完成
void load_model()
{
    FILE *fp;
    int i,j;
    fp = fopen("HiddenWeights.txt", "w");
    
    if(fp==NULL)
    {
        printf("load HiddenWeights fail!\n");
        return;
    }
    for (i = 0; i <= new_length * 3; i++)
    {
        for (j = 0; j < HiddenNodes; j++)
        {
            fprintf(fp, "%f\n", HiddenWeights[i][j]);
        }
    }
    fclose(fp);
    fp = fopen("OutputWeights.txt", "w");
    if(fp==NULL)
    {
        printf("load OutputWeights fail!\n");
        return;
    }
    for (i = 0; i <= HiddenNodes; i++)
    {
        for (j = 0; j < target_num; j++)
        {
            fprintf(fp, "%f\n", OutputWeights[i][j]);
        }
    }
    fclose(fp);
} 



int main ()
{
    int i, j, p, q, r;
    float accuracy=0;
    if(setup())
        return 0;
/******************************************************************
* Initialize HiddenWeights and ChangeHiddenWeights 
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= new_length * 3 ; j++ ) { 
            ChangeHiddenWeights[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
/******************************************************************
* Initialize OutputWeights and ChangeOutputWeights
******************************************************************/

    for( i = 0 ; i < target_num ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;  
            Rando = (float)((rand() % 100))/100;        
            OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
    //printf("\nInitial/Untrained Outputs: ");
    //run_train_data();

/******************************************************************
* Begin training 
******************************************************************/

    for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) 
    {    

/******************************************************************
* Randomize order of training patterns 讓每次的訓練樣本順序隨機
******************************************************************/

        for( p = 0 ; p < train_data_num ; p++) {
            q = rand()%train_data_num;
            r = RandomizedIndex[p] ; 
            RandomizedIndex[p] = RandomizedIndex[q] ; 
            RandomizedIndex[q] = r ;
        }
        Error = 0.0 ;
/******************************************************************
* Cycle through each training pattern in the randomized order
******************************************************************/
        for( q = 0 ; q < train_data_num ; q++ ) 
        {    
            p = RandomizedIndex[q];

/******************************************************************
* Compute hidden layer activations
******************************************************************/

            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = HiddenWeights[new_length * 3][i] ;
                for( j = 0 ; j < new_length * 3 ; j++ ) {
                    Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

            for( i = 0 ; i < target_num ; i++ ) {    
                Accum = OutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    Accum += Hidden[j] * OutputWeights[j][i] ;
                }
                Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
                OutputDelta[i] = (train_data_output[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
                Error += 0.5 * (train_data_output[p][i] - Output[i]) * (train_data_output[p][i] - Output[i]) ;
            }

/******************************************************************
* Backpropagate errors to hidden layer
******************************************************************/

            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = 0.0 ;
                for( j = 0 ; j < target_num ; j++ ) {
                    Accum += OutputWeights[i][j] * OutputDelta[j] ;
                }
                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
            }


/******************************************************************
* Update Inner-->Hidden Weights
******************************************************************/


            for( i = 0 ; i < HiddenNodes ; i++ ) {     
                ChangeHiddenWeights[new_length * 3][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[new_length * 3][i] ;
                HiddenWeights[new_length * 3][i] += ChangeHiddenWeights[new_length * 3][i] ;
                for( j = 0 ; j < new_length * 3 ; j++ ) { 
                    ChangeHiddenWeights[j][i] = LearningRate * train_data_input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
                    HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
                }
            }

/******************************************************************
* Update Hidden-->Output Weights
******************************************************************/

            for( i = 0 ; i < target_num ; i ++ ) {    
                ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
                OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
                    OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
                }
            }
        }
        accuracy = Get_Train_Accuracy();

/******************************************************************
* Every 1000 cycles send data to terminal for display
******************************************************************/
        ReportEvery1000 = ReportEvery1000 - 1;
        if (ReportEvery1000 == 0)
        {
            
            printf ("\nTrainingCycle: %ld\n",TrainingCycle);
            printf ("Error = %.5f\n",Error);
            printf ("Accuracy = %.2f /100 \n",accuracy*100);
            //run_train_data();

            if (TrainingCycle==1)
            {
                ReportEvery1000 = 999;
            }
            else
            {
                ReportEvery1000 = 1000;
            }
        }    


/******************************************************************
* If error rate is less than pre-determined threshold then end
******************************************************************/

        if( accuracy >= goal_acc ) break ;  
    }

    printf ("TrainingCycle: %ld\n",TrainingCycle);
    printf ("Error = %.5f\n",Error);
    run_train_data();
    printf ("Training Set Solved!\n ");
    printf ("--------\n"); 
    printf ("Testing Start!\n ");
    run_test_data();
    printf ("--------\n"); 
    ReportEvery1000 = 1;
    load_model();
    return 0;
}
