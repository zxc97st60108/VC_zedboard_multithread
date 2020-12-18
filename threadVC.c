
// copy from dvc_sw_no_terrible_sound.c
float PreMean[] = {-0.09846516, 1.217689964, 2.221828978, 2.386463096, 2.229936436, 2.106549394, 1.956171095, 1.866162951, 1.608338622, 1.201791685, 0.716337163, 0.509326174, 0.393072253, 0.15799105, 0.13177732, 0.262901654, 0.328990115, 0.406902552, 0.394303374, 0.39014851, 0.34884474, 0.336672951, 0.341558047, 0.334856213, 0.27356489, 0.167099009, -0.003567565, -0.181335547, -0.335635616, -0.472672478, -0.629839701, -0.716304766, -0.728442701, -0.722806756, -0.756926182, -0.755569926, -0.725505722, -0.610453495, -0.501662124, -0.396483291, -0.238279715, -0.149759726, -0.147753751, -0.09066521, -0.054220917, -0.07930843, -0.13832354, -0.217686532, -0.274343598, -0.355578915, -0.449888398, -0.47817359, -0.519536185, -0.522493588, -0.522180531, -0.456945968, -0.424724502, -0.377967555, -0.381591213, -0.395609271, -0.443062784, -0.467546859, -0.549417069, -0.607887035, -0.694635952, -0.780200296, -0.901361516, -0.989335228, -1.095202699, -1.211778558, -1.301850907, -1.401107667, -1.476722869, -1.548434873, -1.621917712, -1.69418043, -1.764963542, -1.797895318, -1.848219693, -1.854125838, -1.871111743, -1.882218347, -1.913421106, -1.896030419, -1.907530615, -1.903852197, -1.885308545, -1.889465824, -1.8666847, -1.860385003, -1.834408222, -1.805265764, -1.777318162, -1.749779785, -1.731101729, -1.687532632, -1.681037414, -1.650766923, -1.656911663, -1.639487129, -1.635987002, -1.642502338, -1.65072346, -1.663049413, -1.671323741, -1.690024461, -1.70156977, -1.707826072, -1.72052989, -1.740551021, -1.767264106, -1.758761339, -1.802833693, -1.810275768, -1.844288544, -1.845332518, -1.883473136, -1.917301556, -1.940047911, -1.986665601, -2.033264167, -2.070977902, -2.103420379, -2.15727324, -2.215116775, -2.232971444, -2.276461328, -2.269344902, -2.63086578};
float PreVari[] = {1.359104658, 1.148397731, 1.189749786, 1.248057681, 1.330039782, 1.421857655, 1.518444713, 1.582303281, 1.614599391, 1.567102479, 1.551594764, 1.655935098, 1.649253622, 1.56349393, 1.551557942, 1.59303153, 1.612055249, 1.634315622, 1.649696373, 1.643217639, 1.65295106, 1.649077472, 1.641140731, 1.62711342, 1.60700287, 1.568641949, 1.529489138, 1.485738308, 1.449017882, 1.403722587, 1.372737688, 1.306978876, 1.234319554, 1.243182955, 1.262266571, 1.259727296, 1.291291633, 1.326234541, 1.369549967, 1.409225875, 1.438132283, 1.451832671, 1.477819251, 1.485115196, 1.500326386, 1.500533925, 1.503877343, 1.480382194, 1.443138305, 1.457706095, 1.475407721, 1.458867124, 1.47997716, 1.4781484, 1.484673867, 1.478629136, 1.485450357, 1.467170759, 1.463131184, 1.438374844, 1.428973538, 1.400343781, 1.398743335, 1.357180654, 1.322661803, 1.315693453, 1.319357676, 1.295390198, 1.292724303, 1.284112906, 1.271745948, 1.253770384, 1.252974214, 1.22425075, 1.21030763, 1.198606921, 1.185976635, 1.180311605, 1.191289634, 1.183175526, 1.183413527, 1.186501396, 1.198228771, 1.190459094, 1.195566895, 1.201381185, 1.200184372, 1.19912587, 1.217308601, 1.199752295, 1.197538723, 1.202525753, 1.205198726, 1.207963104, 1.224099361, 1.220679649, 1.234114449, 1.220191578, 1.236008334, 1.222045882, 1.219387352, 1.217797598, 1.218989276, 1.210671955, 1.222013522, 1.205851317, 1.205504773, 1.197848418, 1.197180704, 1.193191954, 1.200435198, 1.188929887, 1.208119067, 1.182977914, 1.190784798, 1.175255494, 1.169551034, 1.160219066, 1.151266597, 1.136787518, 1.145570334, 1.105497081, 1.091740095, 1.069399247, 1.048828772, 1.034558072, 1.039705549, 1.007111427, 1.220952319};
float PostMean[] = {0.609899972, 1.997609215, 2.402899277, 2.402271689, 2.29561653, 2.425386783, 2.221022702, 1.786986336, 1.516516338, 1.401556982, 1.270865477, 1.181292164, 1.092872031, 0.906084648, 0.710497765, 0.677573579, 0.492320927, 0.420512939, 0.381366357, 0.385977841, 0.36731393, 0.290702405, 0.184773776, 0.074150305, -0.046317883, -0.173435401, -0.258300133, -0.284482034, -0.271995183, -0.262457175, -0.307112751, -0.311096528, -0.31782137, -0.291784969, -0.275141929, -0.209344112, -0.151326567, -0.072109776, -0.065827515, -0.089572236, -0.121025004, -0.200837279, -0.329112536, -0.282247428, -0.201875127, -0.134632494, -0.066481832, -0.009426184, 0.028150059, 0.032981531, 0.006838762, 0.006128826, -0.014577009, -0.043331622, -0.102218585, -0.13740999, -0.190846354, -0.21672405, -0.300054671, -0.384954459, -0.505683895, -0.57158958, -0.665986906, -0.680813809, -0.725554441, -0.71228331, -0.742035613, -0.743319021, -0.788418067, -0.846989608, -0.912239402, -0.98582889, -1.059571014, -1.099099639, -1.170609851, -1.22953812, -1.300293627, -1.337960248, -1.398712286, -1.378155454, -1.425014427, -1.423819625, -1.494541661, -1.47922689, -1.489731051, -1.515776634, -1.532215429, -1.566430915, -1.619202922, -1.616926097, -1.636647643, -1.654058898, -1.654090807, -1.668915428, -1.683796706, -1.615704223, -1.669719887, -1.594721462, -1.647760268, -1.610500205, -1.578904143, -1.565495623, -1.547144507, -1.524796275, -1.52005914, -1.484924654, -1.476772907, -1.455826962, -1.45192079, -1.451369234, -1.467059368, -1.411209497, -1.44897995, -1.419314669, -1.482851058, -1.478735284, -1.503889513, -1.542640783, -1.572770684, -1.612161835, -1.676513782, -1.687103464, -1.708840917, -1.745986289, -1.774437445, -1.798770841, -1.830986027, -1.773036272, -2.064682342};
float PostVari[] = {1.365549873, 1.184939632, 1.147282643, 1.217714405, 1.295304436, 1.33530451, 1.335846343, 1.347450196, 1.402080578, 1.456917844, 1.541587164, 1.551578845, 1.499167152, 1.46230581, 1.425748807, 1.433351743, 1.415862938, 1.405724144, 1.428433176, 1.415997895, 1.425798842, 1.417744164, 1.400291245, 1.358618587, 1.312882097, 1.217008825, 1.16593442, 1.137073707, 1.137347539, 1.131114806, 1.134714879, 1.10650286, 1.073062871, 1.092870601, 1.136213905, 1.140816282, 1.158670453, 1.149899796, 1.150584432, 1.145492836, 1.145455162, 1.12617365, 1.141560317, 1.153975051, 1.194519155, 1.228174957, 1.259209711, 1.266464709, 1.249691857, 1.273869146, 1.302406129, 1.29814189, 1.305713961, 1.29441044, 1.284307119, 1.266194166, 1.274486226, 1.25663131, 1.245567759, 1.208750508, 1.175932329, 1.135859767, 1.112699399, 1.065339402, 1.04899105, 1.052478926, 1.080575165, 1.078243193, 1.09606683, 1.098924127, 1.105123697, 1.095286774, 1.10891882, 1.081452202, 1.078217354, 1.056962979, 1.050024616, 1.035062616, 1.036813391, 1.014169085, 1.010213081, 1.001585475, 1.012267233, 1.009435604, 1.017207391, 1.01119565, 1.017818493, 1.005351164, 1.037241105, 0.988189174, 0.991870442, 0.984092148, 0.994506196, 0.992587771, 1.006829731, 1.002653578, 1.041098789, 1.019033775, 1.037620687, 1.036861563, 1.039578708, 1.032526534, 1.039931717, 1.026913802, 1.049501624, 1.016258882, 1.023526897, 1.011794602, 1.020069652, 1.012618555, 1.027412275, 1.013761752, 1.038511232, 1.02444597, 1.038539569, 1.031125466, 1.034835456, 1.02713705, 1.024521472, 1.008371916, 1.030361157, 0.986836491, 0.982973013, 0.959617007, 0.950007743, 0.940009512, 0.944614002, 0.925990892, 1.148681707};

// 20190416 將 clear buffer 的動作移到 detect button-push 時執行，避免重複按壓造成聲音重疊
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sched.h>
#include <time.h>
#include <stdbool.h>

#include <pthread.h>   //include pthread API lib
#include <semaphore.h> //include semaphore lib

// Global semaphore
sem_t sem_cvtor, sem_plyer; //宣告 semaphore

#define INPUT_DIM 129
#define OUTPUT_DIM 129
#define HIDDEN1_DIM 512
#define HIDDEN2_DIM 512
#define HIDDEN3_DIM 512
#define MAX_DIM 512
#define NUM_OF_LAYER 4
#define CUTOFF 8000       //    16K / 2
#define SAMPLE_RATE 24000 //    48K / 2
#define FLAG 1
// #define paramPath "./newparam.txt" //你的param放在這
#define paramPath "./model/parameter.txt"

const int frameSize = 256;
const int frameRate = 128;
const int fftSize = 256;
const float pi = 3.14159265358979323846;
const float eps = 0.00000000000000022204;

int DIM[] = {129, 512, 512, 512, 129};
int dimOfmgnitd = 129;
float weight[NUM_OF_LAYER][MAX_DIM][MAX_DIM];
float bias[NUM_OF_LAYER][MAX_DIM];

typedef struct _audio
{
    unsigned char hder[44];
    unsigned short chnl;
    unsigned int fs;
    unsigned int nsmpl;
    unsigned int nframe;
    int dnnframe_idx, pbframe_idx; //check frame finish or not
    int finish_rec;                //check recording finish or not
    float *smpl;
    float *oupt;
    float *hammingWndw;
    float *mgnitd;
    float *phase;
    float *dnnParam;
    float *preMean;
    float *postMean;
    float *preVari;
    float *postVari;
    unsigned int *bit_rv; //32bit*256

} _audio;

typedef struct cplx
{
    float rl;
    float img;

} cplx;

//釋放資源
void garbage_clean(_audio *thisAudio)
{
    free(thisAudio->smpl);
    free(thisAudio->oupt);
}

//錄音function，在這裡設定錄音的功能
void pthread_wav_reader(void *pstruct)
{
    //bind this func to cpu 0
    cpu_set_t set, get;
    int cpu_num = sysconf(_SC_NPROCESSORS_CONF);
    printf("Total %d cpu.\r\n", cpu_num);

    CPU_ZERO(&set);
    CPU_SET(0, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) < 0)
    {
        perror("set thread affinity failed");
        return;
    }
    //check is it run in the processor we give
    CPU_ZERO(&get);
    if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0)
    {
        perror("get thread affinity failed");
        return;
    }
    if (CPU_ISSET(0, &get))
    { //it should be here
        printf("Record thread is running in processor 0\r\n");
    }
    if (CPU_ISSET(1, &get))
    {
        printf("Record thread is running in processor 1\r\n");
    }

    //Start the thread

    const unsigned int mtimeGap = 10000; // 0.1 second
    int idx_i;
    _audio *thisAudio = (_audio *)pstruct;

    FILE *fpr = fopen("/dev/xillybus_audio", "r");
    int btnfd = open("/sys/class/gpio/gpio50/value", O_RDONLY | O_NONBLOCK);

    assert(fpr != NULL);
    assert(btnfd > 0);

    float RC = 1.0 / (CUTOFF * 2 * 3.1415);
    float dt = 1.0 / SAMPLE_RATE;
    float alpha = dt / (RC + dt);
    float filter, old_filter;
    int btnvalue, old_btnvalue, samples;
    short audio_in;

    btnvalue = old_btnvalue = 0;
    // printf("into while \n");
    while (1)
    {
        // busy polling the button value
        read(btnfd, &btnvalue, 1);
        lseek(btnfd, 0, SEEK_SET);
        // btnvalue &= 1;
        btnvalue = btnvalue & 1 ? 0 : 1;

        if (btnvalue)
        { //press down button
            if (!old_btnvalue)
            {
                printf("Start Record\r\n");
                thisAudio->dnnframe_idx = thisAudio->pbframe_idx = 0;
                thisAudio->nsmpl = thisAudio->nframe = 0;
                thisAudio->finish_rec = 0;
                for (idx_i = 0; idx_i < 1000000; idx_i++)
                {
                    thisAudio->oupt[idx_i] = thisAudio->smpl[idx_i] = 0;
                }
                samples = 0;
                printf("end Record\r\n");
            }
            old_btnvalue = btnvalue;

            // printf("1  \r\n");
            //get two channel audio input
            fread(&audio_in, 2, 1, fpr);
            fread(&audio_in, 2, 1, fpr);

            if (samples == 0)
            {
                filter = old_filter = (float)audio_in;
            }
            else
            {
                // printf("2  \r\n");
                //filter for 48KHz to 16KHz
                filter = old_filter + (alpha * ((float)audio_in) - old_filter);
                old_filter = filter;
                // no filter
                filter = audio_in;

                if (samples % 3 == 0) //down sampling to 16KHz
                {
                    // printf("3  \r\n");
                    //注意在此是將 sample 存進 thisAudio->smpl 中，並同時進行ZeroScore Norm正規化處理
                    thisAudio->smpl[thisAudio->nsmpl] = ((float)filter / 32768 + 0.000464) / 0.04953575;
                    // printf(" smpl = %f \n", thisAudio->smpl[thisAudio->nsmpl]);
                    //-------------------------------------------------------------------------------//
                    //0.000172 is mean on zb                                                         //
                    //0.0111 is std on zb                                                            //
                    //找到你自己音訊的 mean 以及 std 參數，可以利用你之前VC程式中的 ZeroScoreNorm 函式來找//
                    //-------------------------------------------------------------------------------//

                    // printf("3-1  \r\n");
                    //當按著按鈕錄音時，持續根據你錄製的音訊長度增加frame的數目
                    thisAudio->nsmpl++;
                    //-------------------------------------------------------------------------------//
                    // 設定若錄到的 sample 數能夠滿足一筆 frame 的大小時，就發送訊號給轉換的 function     //
                    // printf("nsmpl = %d \n", thisAudio->nsmpl);
                    // printf("nframe = %d \n", thisAudio->nframe);
                    // printf("3-2  \r\n");
                    if (thisAudio->nsmpl >= frameSize && thisAudio->nsmpl % frameSize == 0)
                    { //判斷錄製的sample數       //

                        // printf("3-2-5  \r\n"); //frame counter ++
                        thisAudio->nframe++;
                        sem_post(&sem_cvtor); //發送訊號給下一個 function 進行轉換
                                              // printf("3-3  \r\n"); //frame counter ++                                        //
                    }
                    // printf("3-4  \r\n"); //frame counter ++           //
                    //-------------------------------------------------------------------------------//
                    // printf("4  \r\n");
                }
                // printf("5  \r\n");
            }

            // printf("6  \r\n");
            samples++;
        }
        else
        { //idle && release status

            if (old_btnvalue)
            { //release the button
                thisAudio->finish_rec = 1;
                printf("Finish record thread with %d frames!\r\n", thisAudio->nframe);
                //break;
            }

            old_btnvalue = btnvalue;
        }
    }

    fclose(fpr);
    close(btnfd);

    pthread_exit(NULL);
}

void FFT(_audio *thisAudio)
{
    int stage, num_block, num_bf2, start, jump, tw_idx;
    int dimOfmgnitd = frameSize / 2 + 1;
    cplx data[fftSize];
    cplx t;
    bool is_change[256] = {};
    stage = 8;
    num_bf2 = 128;
    num_block = 1;
    jump = fftSize;
    //hamming window
    for (int idx_i = thisAudio->dnnframe_idx * frameRate, idx_j = 0; idx_j < frameSize; idx_i++, idx_j++)
    {
        // printf("hammingWindow = %f , sample = %f \n", thisAudio->hammingWndw[idx_j], thisAudio->smpl[idx_i]);
        data[idx_j].rl = thisAudio->hammingWndw[idx_j] * thisAudio->smpl[idx_i];
        data[idx_j].img = 0;
    }

    //FFT
    //////////////////////////////////
    //input : real 256 img 256 (0)
    //output : real 129 img 129
    //write ur code here//
    //////////////////////////////////
    for (int idx_i = 0; idx_i < stage; idx_i++)
    {
        // printf("[num_bf2,num_block]=[%d,%d]\n", num_bf2, num_block);
        start = 0;
        for (int idx_j = 0; idx_j < num_block; idx_j++)
        {
            tw_idx = 0;
            for (int idx_k = start; idx_k < (num_bf2 + start); idx_k++)
            {
                t.rl = data[idx_k + num_bf2].rl;
                t.img = data[idx_k + num_bf2].img;
                data[idx_k + num_bf2].rl = data[idx_k].rl - t.rl;
                data[idx_k + num_bf2].img = data[idx_k].img - t.img;
                data[idx_k].rl += t.rl;
                data[idx_k].img += t.img;
                t.rl = data[idx_k + num_bf2].rl;
                t.img = data[idx_k + num_bf2].img;
                data[idx_k + num_bf2].rl = t.rl * cos((2 * 3.1415926 * tw_idx) / fftSize) - t.img * -sin((2 * 3.1415926 * tw_idx) / fftSize);
                data[idx_k + num_bf2].img = t.img * cos((2 * 3.1415926 * tw_idx) / fftSize) + t.rl * -sin((2 * 3.1415926 * tw_idx) / fftSize);
                // printf("[%d,%d,%d]\n", k, k + num_bf2, tw_idx);
                tw_idx += num_block;
            }
            start += jump;
        }
        jump = jump >> 1;
        num_bf2 = num_bf2 >> 1;
        num_block = num_block << 1;
    }

    int idx_j;
    // Bit reverse
    for (int idx_i = 0; idx_i < fftSize; idx_i++)
    {
        if (!is_change[idx_i])
        {
            idx_j = thisAudio->bit_rv[idx_i];
            t.rl = data[idx_i].rl;
            t.img = data[idx_i].img;
            data[idx_i].rl = data[idx_j].rl;
            data[idx_i].img = data[idx_j].img;
            data[idx_j].rl = t.rl;
            data[idx_j].img = t.img;
            is_change[idx_i] = 1;
            is_change[idx_j] = 1;
        }
    }

    //phase = atan2f(img,rl)
    //////////////////////////////////
    //input real 129 img 129
    //output phase 129
    //write ur code here//
    //////////////////////////////////
    for (int idx_i = 0; idx_i < frameSize; idx_i++)
    {
        thisAudio->phase[idx_i] = atan2f(data[idx_i].img, data[idx_i].rl);
    }

    for (int idx_i = 0; idx_i < dimOfmgnitd; idx_i++)
    {
        thisAudio->mgnitd[idx_i] = data[idx_i].rl * data[idx_i].rl + data[idx_i].img * data[idx_i].img;
        if (thisAudio->mgnitd[idx_i] == 0)
        {
            thisAudio->mgnitd[idx_i] = eps;
        }
    }
}

void preNorm(_audio *thisAudio)
{
    int dimOfmgnitd = frameSize / 2 + 1;
    //log
    //////////////////////////////////
    //input magnitd 129
    //output magnitd 129(log)
    //write ur code here//
    //////////////////////////////////

    //norm_T
    //////////////////////////////////
    //input magnitd 129
    //output magnitd 129(norm)
    //write ur code here//
    //////////////////////////////////
    for (int idx_i = 0; idx_i < dimOfmgnitd; idx_i++)
    {

        thisAudio->mgnitd[idx_i] = log10(thisAudio->mgnitd[idx_i]);
        //thisAudio->mgnitd[idx_i] = (thisAudio->mgnitd[idx_i] - thisAudio->preMean[idx_i]) / thisAudio->preVari[idx_i];
    }
}

void DNN(_audio *thisAudio)
{

    //DNN
    //////////////////////////////////
    /// DNN model 129 * 3_512 * 129 ///
    //input : 129 mgnitd
    //output : 129mgnitd
    // write u code here ///
    //////////////////////////////////
    float v_hidden0[129];
    float v_hidden1[512];
    float v_hidden2[512];
    float v_hidden3[512];
    float v_output[129];
    // inference of first hidden layer
    for (int i = 0; i < 512; i++)
    {
        v_hidden1[i] = bias[0][i];
        for (int j = 0; j < 129; j++)
        {
            v_hidden1[i] += thisAudio->mgnitd[j] * weight[0][i][j];
        }
        v_hidden1[i] = v_hidden1[i] > 0 ? v_hidden1[i] : 0;
    }
    //
    for (int i = 0; i < 512; i++)
    {
        v_hidden2[i] = bias[1][i];
        for (int j = 0; j < 512; j++)
        {
            v_hidden2[i] += v_hidden1[j] * weight[1][i][j];
        }
        v_hidden2[i] = v_hidden2[i] > 0 ? v_hidden2[i] : 0;
    }

    for (int i = 0; i < 512; i++)
    {
        v_hidden3[i] = bias[2][i];
        for (int j = 0; j < 512; j++)
        {
            v_hidden3[i] += v_hidden2[j] * weight[2][i][j];
        }
        v_hidden3[i] = v_hidden3[i] > 0 ? v_hidden3[i] : 0;
    }

    //
    // inference of output layer
    for (int i = 0; i < 129; i++)
    {
        v_output[i] = bias[3][i];
        for (int j = 0; j < 512; j++)
        {
            v_output[i] += v_hidden3[j] * weight[3][i][j];
        }
        thisAudio->mgnitd[i] = v_output[i];
        // printf("vout = %f ", v_output[i]);
    }
}

void postNorm(_audio *thisAudio)
{
    //Denorm_T
    //////////////////////////////////
    //input magnitd 129
    //output magnitd 129(Denorm)
    //write ur code here//
    //////////////////////////////////

    // pow(10,x)
    //////////////////////////////////
    //input magnitd 129 (Denorm)
    //output magnitd 129 (10^Denorm)
    //write ur code here//
    //////////////////////////////////
    for (int idx_i = 0; idx_i < 129; idx_i++)
    {
        //thisAudio->mgnitd[idx_i] = thisAudio->mgnitd[idx_i] * thisAudio->postVari[idx_i] + thisAudio->postMean[idx_i];
        thisAudio->mgnitd[idx_i] = pow(10, thisAudio->mgnitd[idx_i]);
    }
}

void IFFT(_audio *thisAudio)
{
    cplx data[fftSize], out[fftSize + 1];

    // the following for loop was originally inside the postNorm() function
    for (int idx_i = 0; idx_i < 129; idx_i++)
    {
        thisAudio->mgnitd[idx_i] = sqrtf(thisAudio->mgnitd[idx_i]);
    }

    for (int idx_i = 1; idx_i < 130; idx_i++)
    {
        out[idx_i].rl = cosf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
        out[idx_i].img = sinf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
        // out[idx_i].rl = cosf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
        // out[idx_i].img = sinf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
    }
    for (int idx_i = 1; idx_i < 128; idx_i++)
    {
        // out[idx_i + 129] = conjf(out[129 - idx_i]); //共軛
        out[idx_i + 129].rl = out[129 - idx_i].rl;
        out[idx_i + 129].img = -out[129 - idx_i].img;
    }
    for (int idx_i = 0; idx_i < fftSize; idx_i++)
    {
        data[idx_i].rl = out[idx_i + 1].rl;
        data[idx_i].img = out[idx_i + 1].img;
        // xin[idx_i].img = out[idx_i + 1].img;
    }
    //IFFT part
    int stage, num_block, num_bf2, start, jump, tw_idx;
    cplx t;
    bool is_change[256] = {};
    stage = 8;
    num_bf2 = 128;
    num_block = 1;
    jump = fftSize;

    //FFT
    for (int idx_i = 0; idx_i < stage; idx_i++)
    {
        // printf("[num_bf2,num_block]=[%d,%d]\n", num_bf2, num_block);
        start = 0;
        for (int idx_j = 0; idx_j < num_block; idx_j++)
        {
            tw_idx = 0;
            for (int idx_k = start; idx_k < (num_bf2 + start); idx_k++)
            {
                t.rl = data[idx_k + num_bf2].rl;
                t.img = data[idx_k + num_bf2].img;
                data[idx_k + num_bf2].rl = data[idx_k].rl - t.rl;
                data[idx_k + num_bf2].img = data[idx_k].img - t.img;
                data[idx_k].rl += t.rl;
                data[idx_k].img += t.img;
                t.rl = data[idx_k + num_bf2].rl;
                t.img = data[idx_k + num_bf2].img;
                data[idx_k + num_bf2].rl = t.rl * cos((2 * 3.1415926 * tw_idx) / fftSize) - t.img * sin((2 * 3.1415926 * tw_idx) / fftSize);
                data[idx_k + num_bf2].img = t.img * cos((2 * 3.1415926 * tw_idx) / fftSize) + t.rl * sin((2 * 3.1415926 * tw_idx) / fftSize);
                // printf("[%d,%d,%d]\n", k, k + num_bf2, tw_idx);
                tw_idx += num_block;
            }
            start += jump;
        }
        jump = jump >> 1;
        num_bf2 = num_bf2 >> 1;
        num_block = num_block << 1;
    }

    int idx_j;
    // Bit reverse
    for (int idx_i = 0; idx_i < fftSize; idx_i++)
    {
        if (!is_change[idx_i])
        {
            idx_j = thisAudio->bit_rv[idx_i];
            t.rl = data[idx_i].rl;
            t.img = data[idx_i].img;
            data[idx_i].rl = data[idx_j].rl;
            data[idx_i].img = data[idx_j].img;
            data[idx_j].rl = t.rl;
            data[idx_j].img = t.img;
            is_change[idx_i] = 1;
            is_change[idx_j] = 1;
        }
    }
    // write back cv result to oupt array
    for (int idx_i = thisAudio->dnnframe_idx * frameRate, idx_j = 0; idx_j < fftSize; idx_i++, idx_j++)
    {
        thisAudio->oupt[idx_i] += data[idx_j].rl;
        // printf("oupt = %f \n", thisAudio->oupt[idx_i]);
    }
}
//轉換function，你要在裡面加上你的paramInit、FFT、DNN、IFFT運算，如果有要額外宣告function也記得加上去
void pthread_wav_convertor(void *pstruct)
{
    //bind DVC func to cpu 1
    cpu_set_t set, get;

    CPU_ZERO(&set);
    CPU_SET(1, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) < 0)
    {
        perror("set thread affinity failed");
        return;
    }
    //check is it run in the processor we give
    CPU_ZERO(&get);
    if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0)
    {
        perror("get thread affinity failed");
        return;
    }
    if (CPU_ISSET(0, &get))
    {
        printf("DVC thread is running in processor 0\r\n");
    }
    if (CPU_ISSET(1, &get))
    { //it should be here
        printf("DVC thread is running in processor 1\r\n");
    }

    //Start the thread
    _audio *thisAudio = (_audio *)pstruct;
    int idx_i, nframe;
    // relative settings
    // refer to function frame_based_cv()
    //-------------------------------------------------------------------------------//
    // 在進行運算前初始化，在這裡加上你的parameterInit function                         //
    //-------------------------------------------------------------------------------//
    // printf("parameterInit \n");
    // int idx_i;
    float temp;
    int featLen = fftSize / 2 + 1;
    thisAudio->hammingWndw = (float *)malloc(sizeof(float) * frameSize);
    thisAudio->mgnitd = (float *)malloc(sizeof(float) * featLen);
    thisAudio->phase = (float *)malloc(sizeof(float) * fftSize);
    thisAudio->preMean = (float *)malloc(sizeof(float) * featLen);
    thisAudio->postMean = (float *)malloc(sizeof(float) * featLen);
    thisAudio->preVari = (float *)malloc(sizeof(float) * featLen);
    thisAudio->postVari = (float *)malloc(sizeof(float) * featLen);
    thisAudio->bit_rv = (int *)malloc(sizeof(int) * fftSize);
    // thisAudio->tw_factor = (complex *)malloc(sizeof(complex) * featLen);
    assert(thisAudio->hammingWndw != NULL && thisAudio->mgnitd != NULL && thisAudio->phase != NULL && thisAudio->preMean != NULL && thisAudio->postMean != NULL && thisAudio->preVari != NULL && thisAudio->postVari != NULL);

    //===fft
    unsigned int bit_rv[] = {0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240, 8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248, 4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252, 2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250, 6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246, 14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254, 1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241, 9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249, 5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245, 13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253, 3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251, 7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247, 15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255};

    for (idx_i = 0; idx_i < fftSize; idx_i++)
        thisAudio->bit_rv[idx_i] = bit_rv[idx_i];

    for (idx_i = 0; idx_i < featLen; idx_i++)
    {
        thisAudio->preMean[idx_i] = PreMean[idx_i];
        thisAudio->postMean[idx_i] = PostMean[idx_i];
        thisAudio->preVari[idx_i] = PreVari[idx_i];
        thisAudio->postVari[idx_i] = PostVari[idx_i];
    }
    // printf("讀取weight & bias參數 \n");
    //-------------------讀取weight & bias參數-------------------//
    // dnn param settings
    const int lenOfDNN = 658049; // 129*512 + 512 + 512*512 + 512 + 512*512 + 512 + 512*129 + 129;
    float *dnn = (float *)malloc(sizeof(float) * lenOfDNN);
    assert(dnn != NULL);

    for (idx_i = 0; idx_i < frameSize; idx_i++)
    {
        thisAudio->hammingWndw[idx_i] = 0.54 - 0.46 * cosf(2.0 * pi * idx_i / (frameSize - 1));
    }

    long long int input_temp;
    int notuse;
    // read param from textfile
    FILE *fp = fopen(paramPath, "r");
    assert(fp != NULL);
    //not usel parameter
    for (int i = 0; i < 6; i++)
        fscanf(fp, "%d", &notuse);

    // for (int i = 0; i < 5; i++)
    // printf("%d\n", DIM[i]);

    for (int i = 0; i < NUM_OF_LAYER; i++)
    {
        // printf("read weight \n");
        //read weight
        for (int j = 0; j < DIM[i + 1]; j++)
            for (int k = 0; k < DIM[i]; k++)
            {
                fscanf(fp, "%llx", &input_temp);
                weight[i][j][k] = *((double *)&input_temp);
                if (i == 0)
                {
                    weight[i][j][k] /= PreVari[k];
                }
                else if (i == (NUM_OF_LAYER - 1))
                {
                    weight[i][j][k] *= PostVari[j];
                }
            }
        // printf("read bias \n");
        //read bias
        for (int j = 0; j < DIM[i + 1]; j++)
        {
            fscanf(fp, "%llx", &input_temp);
            bias[i][j] = *((double *)&input_temp);
            if (i == 0)
            {
                for (int k = 0; k < DIM[i]; k++)
                {
                    bias[i][j] = bias[i][j] - (PreMean[k] * weight[i][j][k]);
                }
            }
            else if (i == (NUM_OF_LAYER - 1))
            {
                bias[i][j] = bias[i][j] * PostVari[j] + PostMean[j];
            }
            // printf("j = %d \n", j);
        }
        // printf("read bias end \n");
    }
    fclose(fp);
    //-----------------------------------------------------------------//

    // printf("start conversion \n");
    // conversion
    while (1)
    {
        if (thisAudio->dnnframe_idx < thisAudio->nframe)
        { //convert status

            //在這接收錄音的semaphore，等待錄音完一筆frame的訊號，當訊號進來時才開始運算
            sem_wait(&sem_cvtor);
            // printf(">>COMPUTING\n");
            //-------------------------------------------------------------------------------//
            // 在這裡是實際進行 FFT -> DNN -> IFFT 運算的部分，其他的 function 前面已經進行過    //
            // FFT(thisAudio);
            int stage, num_block, num_bf2, start, jump, tw_idx;
            // int dimOfmgnitd = frameSize / 2 + 1;
            cplx data[fftSize];
            cplx t;
            bool is_change[256] = {};
            stage = 8;
            num_bf2 = 128;
            num_block = 1;
            jump = fftSize;
            //hamming window
            for (int idx_i = thisAudio->dnnframe_idx * frameRate, idx_j = 0; idx_j < frameSize; idx_i++, idx_j++)
            {
                // printf("hammingWindow = %f , sample = %f \n", thisAudio->hammingWndw[idx_j], thisAudio->smpl[idx_i]);
                data[idx_j].rl = thisAudio->hammingWndw[idx_j] * thisAudio->smpl[idx_i];
                data[idx_j].img = 0;
            }

            for (int idx_i = 0; idx_i < stage; idx_i++)
            {
                // printf("[num_bf2,num_block]=[%d,%d]\n", num_bf2, num_block);
                start = 0;
                for (int idx_j = 0; idx_j < num_block; idx_j++)
                {
                    tw_idx = 0;
                    for (int idx_k = start; idx_k < (num_bf2 + start); idx_k++)
                    {
                        t.rl = data[idx_k + num_bf2].rl;
                        t.img = data[idx_k + num_bf2].img;
                        data[idx_k + num_bf2].rl = data[idx_k].rl - t.rl;
                        data[idx_k + num_bf2].img = data[idx_k].img - t.img;
                        data[idx_k].rl += t.rl;
                        data[idx_k].img += t.img;
                        t.rl = data[idx_k + num_bf2].rl;
                        t.img = data[idx_k + num_bf2].img;
                        data[idx_k + num_bf2].rl = t.rl * cos((2 * 3.1415926 * tw_idx) / fftSize) - t.img * -sin((2 * 3.1415926 * tw_idx) / fftSize);
                        data[idx_k + num_bf2].img = t.img * cos((2 * 3.1415926 * tw_idx) / fftSize) + t.rl * -sin((2 * 3.1415926 * tw_idx) / fftSize);
                        // printf("[%d,%d,%d]\n", k, k + num_bf2, tw_idx);
                        tw_idx += num_block;
                    }
                    start += jump;
                }
                jump = jump >> 1;
                num_bf2 = num_bf2 >> 1;
                num_block = num_block << 1;
            }

            int idx_j;
            // Bit reverse
            for (int idx_i = 0; idx_i < fftSize; idx_i++)
            {
                if (!is_change[idx_i])
                {
                    idx_j = thisAudio->bit_rv[idx_i];
                    t.rl = data[idx_i].rl;
                    t.img = data[idx_i].img;
                    data[idx_i].rl = data[idx_j].rl;
                    data[idx_i].img = data[idx_j].img;
                    data[idx_j].rl = t.rl;
                    data[idx_j].img = t.img;
                    is_change[idx_i] = 1;
                    is_change[idx_j] = 1;
                }
            }

            for (int idx_i = 0; idx_i < frameSize; idx_i++)
            {
                thisAudio->phase[idx_i] = atan2f(data[idx_i].img, data[idx_i].rl);
            }

            for (int idx_i = 0; idx_i < dimOfmgnitd; idx_i++)
            {
                thisAudio->mgnitd[idx_i] = data[idx_i].rl * data[idx_i].rl + data[idx_i].img * data[idx_i].img;
                if (thisAudio->mgnitd[idx_i] == 0)
                {
                    thisAudio->mgnitd[idx_i] = eps;
                }
            }
            // printf(">>FFT\n");
            // preNorm(thisAudio);
            for (int idx_i = 0; idx_i < dimOfmgnitd; idx_i++)
            {
                thisAudio->mgnitd[idx_i] = log10(thisAudio->mgnitd[idx_i]);
            }
            // DNN(thisAudio);

            float v_hidden0[129];
            float v_hidden1[512];
            float v_hidden2[512];
            float v_hidden3[512];
            float v_output[129];
            // inference of first hidden layer
            for (int i = 0; i < 512; i++)
            {
                v_hidden1[i] = bias[0][i];
                for (int j = 0; j < 129; j++)
                {
                    v_hidden1[i] += thisAudio->mgnitd[j] * weight[0][i][j];
                }
                v_hidden1[i] = v_hidden1[i] > 0 ? v_hidden1[i] : 0;
            }
            //
            for (int i = 0; i < 512; i++)
            {
                v_hidden2[i] = bias[1][i];
                for (int j = 0; j < 512; j++)
                {
                    v_hidden2[i] += v_hidden1[j] * weight[1][i][j];
                }
                v_hidden2[i] = v_hidden2[i] > 0 ? v_hidden2[i] : 0;
            }

            for (int i = 0; i < 512; i++)
            {
                v_hidden3[i] = bias[2][i];
                for (int j = 0; j < 512; j++)
                {
                    v_hidden3[i] += v_hidden2[j] * weight[2][i][j];
                }
                v_hidden3[i] = v_hidden3[i] > 0 ? v_hidden3[i] : 0;
            }

            //
            // inference of output layer
            for (int i = 0; i < 129; i++)
            {
                v_output[i] = bias[3][i];
                for (int j = 0; j < 512; j++)
                {
                    v_output[i] += v_hidden3[j] * weight[3][i][j];
                }
                thisAudio->mgnitd[i] = v_output[i];
                // printf("vout = %f ", v_output[i]);
            }
            // printf(">>DNN\n");
            // postNorm(thisAudio);
            for (int idx_i = 0; idx_i < 129; idx_i++)
            {
                thisAudio->mgnitd[idx_i] = pow(10, thisAudio->mgnitd[idx_i]);
            }

            // IFFT(thisAudio);
            cplx data2[fftSize], out[fftSize + 1];

            // the following for loop was originally inside the postNorm() function
            for (int idx_i = 0; idx_i < 129; idx_i++)
            {
                thisAudio->mgnitd[idx_i] = sqrtf(thisAudio->mgnitd[idx_i]);
            }

            for (int idx_i = 1; idx_i < 130; idx_i++)
            {
                out[idx_i].rl = cosf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
                out[idx_i].img = sinf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
                // out[idx_i].rl = cosf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
                // out[idx_i].img = sinf(thisAudio->phase[idx_i - 1]) * thisAudio->mgnitd[idx_i - 1];
            }
            for (int idx_i = 1; idx_i < 128; idx_i++)
            {
                // out[idx_i + 129] = conjf(out[129 - idx_i]); //共軛
                out[idx_i + 129].rl = out[129 - idx_i].rl;
                out[idx_i + 129].img = -out[129 - idx_i].img;
            }
            for (int idx_i = 0; idx_i < fftSize; idx_i++)
            {
                data2[idx_i].rl = out[idx_i + 1].rl;
                data2[idx_i].img = out[idx_i + 1].img;
                // xin[idx_i].img = out[idx_i + 1].img;
            }
            //IFFT part
            // int stage, num_block, num_bf2, start, jump, tw_idx;
            // cplx t;
            bool is_change2[256] = {};
            stage = 8;
            num_bf2 = 128;
            num_block = 1;
            jump = fftSize;

            //FFT
            for (int idx_i = 0; idx_i < stage; idx_i++)
            {
                // printf("[num_bf2,num_block]=[%d,%d]\n", num_bf2, num_block);
                start = 0;
                for (int idx_j = 0; idx_j < num_block; idx_j++)
                {
                    tw_idx = 0;
                    for (int idx_k = start; idx_k < (num_bf2 + start); idx_k++)
                    {
                        t.rl = data2[idx_k + num_bf2].rl;
                        t.img = data2[idx_k + num_bf2].img;
                        data2[idx_k + num_bf2].rl = data2[idx_k].rl - t.rl;
                        data2[idx_k + num_bf2].img = data2[idx_k].img - t.img;
                        data2[idx_k].rl += t.rl;
                        data2[idx_k].img += t.img;
                        t.rl = data2[idx_k + num_bf2].rl;
                        t.img = data2[idx_k + num_bf2].img;
                        data2[idx_k + num_bf2].rl = t.rl * cos((2 * 3.1415926 * tw_idx) / fftSize) - t.img * sin((2 * 3.1415926 * tw_idx) / fftSize);
                        data2[idx_k + num_bf2].img = t.img * cos((2 * 3.1415926 * tw_idx) / fftSize) + t.rl * sin((2 * 3.1415926 * tw_idx) / fftSize);
                        // printf("[%d,%d,%d]\n", k, k + num_bf2, tw_idx);
                        tw_idx += num_block;
                    }
                    start += jump;
                }
                jump = jump >> 1;
                num_bf2 = num_bf2 >> 1;
                num_block = num_block << 1;
            }

            // int idx_j;
            // Bit reverse
            for (int idx_i = 0; idx_i < fftSize; idx_i++)
            {
                if (!is_change2[idx_i])
                {
                    idx_j = thisAudio->bit_rv[idx_i];
                    t.rl = data2[idx_i].rl;
                    t.img = data2[idx_i].img;
                    data2[idx_i].rl = data2[idx_j].rl;
                    data2[idx_i].img = data2[idx_j].img;
                    data2[idx_j].rl = t.rl;
                    data2[idx_j].img = t.img;
                    is_change2[idx_i] = 1;
                    is_change2[idx_j] = 1;
                }
            }
            // write back cv result to oupt array
            for (int idx_i = thisAudio->dnnframe_idx * frameRate, idx_j = 0; idx_j < fftSize; idx_i++, idx_j++)
            {
                thisAudio->oupt[idx_i] += data2[idx_j].rl;
                // printf("oupt = %f \n", thisAudio->oupt[idx_i]);
            }
            // printf(">>IFFT\n");
            // 在這裡加上你的 FFT -> DNN -> IFFT function 來完成 streaming 運算                //
            // 要記住每次只計算一筆 frame 的資料量                                             //
            thisAudio->dnnframe_idx++; //當算完時frame counter++
            // printf(">>dnnframe_idx = %d \n", thisAudio->dnnframe_idx); //
            //-------------------------------------------------------------------------------//
            sem_post(&sem_plyer);
            // printf("sem_plyer = %d \n", semepore);
            // sem_post(&sem_cvtor);
            //發送semaphore給第三個播音function，他會持續等待到所有frame都算完才開始
        }
        if (thisAudio->dnnframe_idx >= thisAudio->nframe && thisAudio->dnnframe_idx != 0)
        {
            //finish or idle
            // printf("finish or idle ,thisAudio->dnnframe_idx = %d \n", thisAudio->dnnframe_idx);
        }
    }

    pthread_exit(NULL);
}

//播音function，這個其實不需要調整
void pthread_wav_player(void *pstruct)
{
    int excd1cnt = 0;
    //bind playback func to cpu 0
    cpu_set_t set;
    cpu_set_t get;

    CPU_ZERO(&set);
    CPU_SET(0, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) < 0)
    {
        perror("set thread affinity failed");
        return;
    }

    CPU_ZERO(&get);
    if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0)
    {
        perror("get thread affinity failed");
        return;
    }
    if (CPU_ISSET(0, &get))
    { //it should be here
        printf("Playback thread is running in processor 0\r\n");
    }
    if (CPU_ISSET(1, &get))
    {
        printf("Playback thread is running in processor 1\r\n");
    }

    //Start the thread
    // printf("_1  \r\n");
    int idx_i;
    _audio *thisAudio = (_audio *)pstruct;
    FILE *fpw = fopen("/dev/xillybus_audio", "w");

    while (1)
    {
        // printf("_2  \r\n");
        if (thisAudio->dnnframe_idx == 0 || !thisAudio->finish_rec || (thisAudio->finish_rec && thisAudio->dnnframe_idx <= thisAudio->nframe / 2 + 30))
        { //idle
            // printf("_2-2  \r\n");
            continue; // 當運算和錄音完成才開始進到下方的 function
        }
        else if (thisAudio->pbframe_idx < thisAudio->nframe)
        { //playback

            // printf("____playback  \r\n");
            sem_wait(&sem_plyer);
            //play one frame

            for (idx_i = thisAudio->pbframe_idx * 128; idx_i < thisAudio->pbframe_idx * 128 + frameRate; idx_i++)
            {

                // printf("oupt = %f \n", thisAudio->oupt[idx_i]);
                float ftemp = thisAudio->oupt[idx_i]; // 9.5
                ftemp = ftemp / 80;

                if (ftemp >= 1 || ftemp <= -1)
                {
                    excd1cnt++;

                    continue;
                }
                // ftemp = ftemp >= 1 || ftemp < -1 ? 0 : ftemp;
                ftemp *= 32767;
                short temp = ftemp;
                short stemp = 0;
#if FLAG == 0
                fwrite(&temp, 2, 2, fpw);
                fwrite(&temp, 2, 2, fpw);
                fwrite(&temp, 2, 2, fpw);
#elif FLAG == 1
                fwrite(&temp, 2, 1, fpw);
                fwrite(&stemp, 2, 1, fpw);
                fwrite(&temp, 2, 1, fpw);
                fwrite(&stemp, 2, 1, fpw);
                fwrite(&temp, 2, 1, fpw);
                fwrite(&stemp, 2, 1, fpw);

#endif
            }
            thisAudio->pbframe_idx++;
        }
        else
        { //finish all process
            printf("finish all process  \r\n");
            time_t current_time;
            int fd_wav;

            time(&current_time);
            char *p = ctime(&current_time);
            int slen = strlen(p), i;
            for (i = 0; i < slen; i++)
            {
                p[i] = (p[i] == ' ' || p[i] == ':') ? '_' : p[i];
            }
            p[slen - 1] = '\0';
            char fname[64];

            printf("Finish playback\r\n");
            printf("abs val exceed 1 cnt:%d\r\n", excd1cnt);
            excd1cnt = 0;
            thisAudio->nframe = thisAudio->nsmpl = 0;
            thisAudio->dnnframe_idx = thisAudio->pbframe_idx = 0;
            thisAudio->finish_rec = 0;

            for (idx_i = 0; idx_i < 1000000; idx_i++)
            {
                thisAudio->oupt[idx_i] = thisAudio->smpl[idx_i] = 0;
            }
        }
        // printf("_end of while  \r\n");
    }

    // printf("_3  \r\n");
    fclose(fpw);

    pthread_exit(NULL);
}

void init(_audio *thisAudio)
{
    // printf("init \n");
    thisAudio->chnl = 1;
    thisAudio->fs = 16000;
    thisAudio->dnnframe_idx = thisAudio->pbframe_idx = 0;
    thisAudio->nframe = thisAudio->nsmpl = 0;
    thisAudio->finish_rec = 0;

    thisAudio->smpl = (float *)malloc(sizeof(float) * 1000000);
    thisAudio->oupt = (float *)malloc(sizeof(float) * 1000000);
}

int main(int argc, char *argv[])
{
    _audio thisAudio;
    init(&thisAudio);

    // parallel begins

    // 初始化 semaphore
    sem_init(&sem_cvtor, 0, 0);
    sem_init(&sem_plyer, 0, 0);

    pthread_t p_reader, p_convertor, p_player; // 3個 function 對應3條 thread
    pthread_create(&p_reader, NULL, (void *)pthread_wav_reader, &thisAudio);
    pthread_create(&p_convertor, NULL, (void *)pthread_wav_convertor, &thisAudio);
    pthread_create(&p_player, NULL, (void *)pthread_wav_player, &thisAudio);
    pthread_join(p_reader, NULL);
    pthread_join(p_convertor, NULL);
    pthread_join(p_player, NULL);
    //-------------------------------------------------------------------------------//
    // 宣告 pthread_create 將程式上方的3個 function 設為 thread                        //
    // 宣告 pthread_join 執行 thread                                                  //
    //-------------------------------------------------------------------------------//

    printf("All threads joined.\r\n");
    garbage_clean(&thisAudio);
    return 0;
}
