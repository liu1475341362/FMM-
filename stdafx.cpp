#include "stdafx.h"

using namespace cv;

#undef CV_MAT_ELEM_PTR_FAST   //将保持已定义状态且在作用域内，直到程序结束或者使用#undef 指令取消定义。


//OpenCV定义的宏来提取象素值
#define CV_MAT_ELEM_PTR_FAST( mat, row, col, pix_size )  \
     ((mat).data.ptr + (size_t)(mat).step*(row) + (pix_size)*(col))


//取得4个数的最小值
inline float
min4(float a, float b, float c, float d)
{
	a = MIN(a, b);
	c = MIN(c, d);
	return MIN(a, c);
}


#define CV_MAT_3COLOR_ELEM(img,type,y,x,c)  CV_MAT_ELEM(img,type,y,(x)*3+(c))

#define KNOWN  0  //known outside narrow band   已知的外狭窄带
#define BAND   1  //narrow band (known)
#define INSIDE 2  //unknown
#define CHANGE 3  //servise


///////////////////////////////////////////////////////////////////////////////////////////
//双向链表中，节点的存储结构，用于下面的“队列优先浮点数”
//
typedef struct HeapElem  
{
	float T;	//记录T
	int i, j;
	struct HeapElem* prev;  //前节点
	struct HeapElem* next;	//后节点
}
HeapElem;


///////////////////////////////////////////////////////////////////////////////////////////
//队列优先浮点数
//
class PriorityQueueFloat   
{
protected:
	HeapElem *mem, *empty, *head, *tail;
	int num=0, in;

public:

	//根据数据大小组织双向链表，此时的链表是空的
	bool Init(const CvMat* f)
	{
		int i, j;

		//求得原始数据的总节点数
		for (i = 0; i < f->rows; i++)
		{
			for (j = 0; j < f->cols; j++)
				num += CV_MAT_ELEM(*f, uchar, i, j) != 0;    //先判断不为空
		}
		if (num <= 0) return false;
		
		//分配内存空间
		mem = (HeapElem*)cvAlloc((num + 2)*sizeof(HeapElem));
		if (mem == NULL) return false;

		//构造头结点
		head = mem;
		head->i = head->j = -1;    //头结点的i,j值设为-1
		head->prev = NULL;
		head->next = mem + 1;
		head->T = -FLT_MAX;  //头节点无限小，之后的节点无限大，方便于入栈的时候判断头结点压入
		empty = mem + 1;	//empty是插入节点后的序列

		//构造接下来的节点
		for (i = 1; i <= num; i++) {
			mem[i].prev = mem + i - 1;
			mem[i].next = mem + i + 1;
			mem[i].i = -1;
			mem[i].T = FLT_MAX;//无限大
		}

		//构造尾节点
		tail = mem + i;
		tail->i = tail->j = -1;
		tail->prev = mem + i - 1;
		tail->next = NULL;
		tail->T = FLT_MAX;//无限大
		return true;
	}

	//利用循环不断添加节点
	bool Add(const CvMat* f) {
		int i, j;
		for (i = 0; i<f->rows; i++) {
			for (j = 0; j<f->cols; j++) {
				if (CV_MAT_ELEM(*f, uchar, i, j) != 0) {
					if (!Push(i, j, 0)) return false;//不断压入节点，如果Push（）成功，返回1，则!Push(）为0，函数终止，并返回FALSE
				}
			}
		}
		return true;
	}

	//单一节点入栈操作
	bool Push(int i, int j, float T) {
		//*tmp:临时存放未插入链表的头结点，empty待插入节点赋予add
		HeapElem *tmp = mem, *add = empty;  
		
		while (tmp->T<T) tmp = tmp->next;//寻找比插入节点T值大的节点
												/*为什么可以用简单的while来作为升序？
												因为一开始，narrowband 的T值都一致，为0（一开始无限大，后来为0），没必要做一个冒泡排
												序来专门排序，在一开始的添加节点过程中以及修复的新像素的T不断的更新，让他与之前的narrowband简
												单相比，插入相应的节点，这，也相当与一个冒泡排序了*/
		//作为T值对应的节点压入
		add->prev = tmp->prev;
		add->next= tmp;
		tmp->prev = add;
		add->next->prev = add;

		add->i = i;
		add->j = j;
		add->T = T;//更新相应的T值
		mem =empty; //更指针mem指向新链表
		in++;
		return true;
	}

	//出栈
	bool Pop(int *i, int *j) {
		//tmp临时存放链表第2个节点
		HeapElem *tmp = head->next;

		*i = tmp->i;
		*j = tmp->j;
		tmp->prev->next = tmp->next;
		tmp->next->prev = tmp->prev;
		tmp->prev = empty->prev;
		tmp->next = empty;
		tmp->prev->next = tmp;
		tmp->next->prev = tmp;
		empty = tmp;
		in--;

		return true;
	}

	//出栈，加入了T
	bool Pop(int *i, int *j, float *T) {
		HeapElem *tmp = head->next;
		if (empty == tmp) return false;
		*i = tmp->i;
		*j = tmp->j;
		*T = tmp->T;
		tmp->prev->next = tmp->next;
		tmp->next->prev = tmp->prev;
		tmp->prev = empty->prev;
		tmp->next = empty;
		tmp->prev->next = tmp;
		tmp->next->prev = tmp;
		empty = tmp;
		in--;
		//      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
		return true;
	}

	PriorityQueueFloat(void) {
		num = in = 0;
		mem = empty = head = tail = NULL;
	}

	~PriorityQueueFloat(void)
	{
		cvFree(&mem);
	}
};


//////////////////////////////////////////////////////////////////////////////////////////

//计算标量积
inline float   //内联函数，节省函数跳转地址运算时间，增大内存开销
VectorScalMult(CvPoint2D32f v1, CvPoint2D32f v2) {       
	return v1.x*v2.x + v1.y*v2.y;
}

//向量长度
inline float
VectorLength(CvPoint2D32f v1) {
	return v1.x*v1.x + v1.y*v1.y;
}

///////////////////////////////////////////////////////////////////////////////////////////
//HEAP::iterator Heap_Iterator;
//HEAP Heap;


//短时距方程，用四个角点方向来更新T值,返回的sol就是T的值，刻画待修复的像素优先性    f:flag矩阵   t:T值矩阵，每个完好的像素都为0（因为cvsub()）。
//	总体上，是以已知像素为靠拢方向，以2个相邻但更不相似的像素方向靠拢			 						而待修补的像素为10^6 。【582】：cvSet(t, cvScalar(1.0e6f, 0, 0, 0));   //将T矩阵中元素每个初始化为值1.0e6f
static float FastMarching_solve(int i1, int j1, int i2, int j2, const CvMat* f, const CvMat* t)
{
	double sol, a11, a22, m12;
	a11 = CV_MAT_ELEM(*t, float, i1, j1);
	a22 = CV_MAT_ELEM(*t, float, i2, j2);
	m12 = MIN(a11, a22);
																	//P1(a11),P2(a22)两点(T值分别T1,T2)
	if (CV_MAT_ELEM(*f, uchar, i1, j1) != INSIDE)					//P1:Band/Known	P2:Inside	T=1+T1
		if (CV_MAT_ELEM(*f, uchar, i2, j2) != INSIDE)
			//如果2个都是完好像素，且这时候P1,P2 的T值不相近
			if (fabs(a11 - a22) >= 1.0)	//求绝对值					//P1:Inside		P2:B/K		T=1+T2    
				sol = 1 + m12;										//
			else	//P1,P2 的T值一样（相近）								//P1:inside		P2:Inside	T=1+min(T1,T2)
				sol = (a11 + a22 + sqrt((double)(2 - (a11 - a22)*(a11 - a22))))*0.5;				//P1:B/K		P2:B/K		T=1+min(T1,T2)   when  |T1-T2|>=1
		else														//							T=(T1+T2+sqrt(2-(T1-T2)^2)/2
			sol = 1 + a11;																
	else if (CV_MAT_ELEM(*f, uchar, i2, j2) != INSIDE)								
		sol = 1 + a22;
	else
		sol = 1 + m12;	//如果这2个像素都是破损像素

	return (float)sol;
}

/////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////
//第三步，图像修复处理
//函数调用myTelea_InpaintFMM(mask, t, output_img, range, Heap);
//t ：存储该像素离到边缘 δΩ的距离值T的矩阵
//
static void   
myTelea_InpaintFMM(const CvMat *mask, CvMat *t, CvMat *out, int range, PriorityQueueFloat *Heap) {
	int i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
	float dist;

   //处理3通道图像，彩色
	if (CV_MAT_CN(out->type) == 3) { 

		while (Heap->Pop(&ii, &jj)) {											//每次取出T值最小的像素  坐标为(ii,jj)
			
			CV_MAT_ELEM(*mask, uchar, ii, jj) = KNOWN;							//标记为known
			for (q = 0; q<4; q++)
			{//循环访问(ii,jj)的四个相邻像素
				if (q == 0) { i = ii - 1; j = jj; }
				 else if (q == 1) { i = ii;   j = jj - 1; }
				  else if (q == 2) { i = ii + 1; j = jj; }
					else if (q == 3) { i = ii;   j = jj + 1; }
				if ((i <= 1) || (j <= 1) || (i> t->rows - 1) || (j> t->cols - 1)) continue;//如果超出待修复图像的边缘，跳过。不是0，和row，是因为这个是扩宽过的mask

				//如果是inside点，围绕该inside点的四个相邻像素修复(i,j)
				if (CV_MAT_ELEM(*mask, uchar, i, j) == INSIDE) {//如果某相邻像素(i,j)为inside
				
					//3个通道各个修复
					for (color = 0; color <= 2; color++) {	  
						CvPoint2D32f gradI, gradT,//gradI:(i,j)亮度梯度  gradT:(i,j)法向量 
									 r; //r:(i,j)与(k,l)差向量
						float Ia = 0, Jx = 0, Jy = 0, s = 0 , w, weight, sat,
								dst, lev, dir;  //距离因子dst,水平集距离因子lev,方向因子dir

				//计算(i,j)的法向量
						//先计算x方向，自然要用y方向的坐标，所以是对j操作
						if (CV_MAT_ELEM(*mask, uchar, i, j + 1) != INSIDE) {         
							if (CV_MAT_ELEM(*mask, uchar, i, j - 1) != INSIDE) {
								gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j + 1) - CV_MAT_ELEM(*t, float, i, j - 1)))*0.5f;
							}
							else {
								//遇到inside，用非inside类与待修像素相减
								gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j + 1) - CV_MAT_ELEM(*t, float, i, j)));
							}
						}
			 			else {
							if (CV_MAT_ELEM(*mask, uchar, i, j - 1) != INSIDE) {
								gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j) - CV_MAT_ELEM(*t, float, i, j - 1)));
							}
							else {
								//2个都为破损像素，值为0
								gradT.x = 0;
							}
						}
						//计算(i,j)y方向上的法向量，对i操作
						if (CV_MAT_ELEM(*mask, uchar, i + 1, j) != INSIDE) {
							if (CV_MAT_ELEM(*mask, uchar, i - 1, j) != INSIDE) {
								gradT.y = (float)((CV_MAT_ELEM(*t, float, i + 1, j) - CV_MAT_ELEM(*t, float, i - 1, j)))*0.5f;
							}
							else {
								//遇到inside，用非inside类与自己相减
								gradT.y = (float)((CV_MAT_ELEM(*t, float, i + 1, j) - CV_MAT_ELEM(*t, float, i, j)));
							}
						}
						else {
							if (CV_MAT_ELEM(*mask, uchar, i - 1, j) != INSIDE) {
								gradT.y = (float)((CV_MAT_ELEM(*t, float, i, j) - CV_MAT_ELEM(*t, float, i - 1, j)));
							}
							else {
								gradT.y = 0;
							}
						}

				//待修复像素(i,j)，分为2个部分，权值和灰度值。修复半径为range 
						//计算半径内的所有像素的权值weight,即是I(p)的分母
						for (int k = i - range; k <= i + range; k++)//相当于以（i，j）为原点的上下2个半圆合起来的整个圆内的所有的点，这个圆如果合适，必然会有known，inside（不算）。
						{       
							int km = k - 1 + (k == 1), kp = k - 1 - (k == t->rows - 2);//规定了这个修复半径在修复的边缘上如何处理，但是还有一种情况，超过了边界，比如-5行
							for (l = j - range; l <= j + range; l++) {
								int lm = l - 1 + (l == 1), lp = l - 1 - (l == t->cols - 2);
								
								//so，这里规定了修复半径不过图像边缘
								if (k>0 && l>0 && k<t->rows - 1 && l<t->cols - 1) {
									if ((CV_MAT_ELEM(*mask, uchar, k, l) != INSIDE) &&      //如果(i,j)邻域内某(k,l)是B/K
										( (k - i)*(k - i)+(l - j)*(l - j)  <= range*range)) {
										//r是（i,j）和（k,l）的向量差
										r.x = (float)(i - k);
										r.y = (float)(j - l);

										//范数1距离的计算，非模长
										dst = (float)(1. / (VectorLength(r)*sqrt((double)VectorLength(r))));   //计算三个权值分量
										lev = (float)(1. / (1 + fabs(CV_MAT_ELEM(*t, float, k, l) - CV_MAT_ELEM(*t, float, i, j))));
										dir = VectorScalMult(r, gradT);//r是（i,j）和（k,l）【点p，q】的向量差，gradT是（i,j）[点p]的法向量

										if (fabs(dir) <= 0.01) dir = 0.000001f;
										weight = (float)fabs(dst*lev*dir);  //该像素点的权值weight，接下来计算该像素点的灰度值

										//计算(i,j)【点p】处亮度梯度，即是根据多处的（k,l）[点q]的亮度来加权重算填充，每点分为x,y方向
										//x方向，if是为了排除破损点
										if (CV_MAT_ELEM(*mask, uchar, k, l + 1) != INSIDE) {     
											if (CV_MAT_ELEM(*mask, uchar, k, l - 1) != INSIDE) {
												gradI.x = (float)((CV_MAT_3COLOR_ELEM(*out, uchar, km, lp + 1, color) - CV_MAT_3COLOR_ELEM(*out, uchar, km, lm - 1, color)))*2.0f;
											}
											else {
												gradI.x = (float)((CV_MAT_3COLOR_ELEM(*out, uchar, km, lp + 1, color) - CV_MAT_3COLOR_ELEM(*out, uchar, km, lm, color)));
											}
										}
										else {
											if (CV_MAT_ELEM(*mask, uchar, k, l - 1) != INSIDE) {
												gradI.x = (float)((CV_MAT_3COLOR_ELEM(*out, uchar, km, lp, color) - CV_MAT_3COLOR_ELEM(*out, uchar, km, lm - 1, color)));
											}
											else {
												gradI.x = 0;//如果左右像素都是inside，也就是待修复像素
											}
										}
										//y方向
										if (CV_MAT_ELEM(*mask, uchar, k + 1, l) != INSIDE) {
											if (CV_MAT_ELEM(*mask, uchar, k - 1, l) != INSIDE) {
												gradI.y = (float)((CV_MAT_3COLOR_ELEM(*out, uchar, kp + 1, lm, color) - CV_MAT_3COLOR_ELEM(*out, uchar, km - 1, lm, color)))*2.0f;
											}
											else {
												gradI.y = (float)((CV_MAT_3COLOR_ELEM(*out, uchar, kp + 1, lm, color) - CV_MAT_3COLOR_ELEM(*out, uchar, km, lm, color)));
											}
										}
										else {
											if (CV_MAT_ELEM(*mask, uchar, k - 1, l) != INSIDE) {
												gradI.y = (float)((CV_MAT_3COLOR_ELEM(*out, uchar, kp, lm, color) - CV_MAT_3COLOR_ELEM(*out, uchar, km - 1, lm, color)));
											}
											else {
												gradI.y = 0;
											}
										}

										//加权和，和体现在2个for循环上
										Ia += (float)weight * (float)(CV_MAT_3COLOR_ELEM(*out, uchar, km, lm, color));//p点的权值*灰度
										Jx -= (float)weight * (float)(gradI.x*r.x);		//Jx是q点在x方向上的加权亮度梯度，初始值为0 ，计算出来是+=的负数.gradI.x是q点的x方向上的亮度梯度，r.x是（p-q）在x方向上的差（方向向量）
										Jy -= (float)weight * (float)(gradI.y*r.y);		//Jy是q点在y方向上的加权亮度梯度，
										s += weight;//所有像素的权值weight,即是I(p)的分母
									}
								}
							}
						}

						//(i,j)的新I值
						sat = (float)((Ia / s + (Jx + Jy) / (sqrt(Jx*Jx + Jy*Jy)/s + 1.0e-20f) + 0.5f));   //具体数值作为偏置项，把数据放在合理的颜色色阶上
						{	//填入相应的修复像素
							CV_MAT_3COLOR_ELEM(*out, uchar, i - 1, j - 1, color) = cv::saturate_cast<uchar>(sat);//saturate_cast防止数据溢出
						}
					}
					//更新T值
					dist = min4(FastMarching_solve(i - 1, j, i, j - 1, mask, t),
						FastMarching_solve(i - 1, j, i, j + 1, mask, t),
						FastMarching_solve(i + 1, j, i, j - 1, mask, t),
						FastMarching_solve(i + 1, j, i, j + 1, mask, t));
					CV_MAT_ELEM(*t, float, i, j) = dist;
					
					CV_MAT_ELEM(*mask, uchar, i, j) = BAND;   //将(i,j)标记为band
					Heap->Push(i, j, dist);				   //存入优先队列
				}
			}
		}

	}
	
	//处理单通道图像
	else if (CV_MAT_CN(out->type) == 1) {   

		while (Heap->Pop(&ii, &jj)) {

			CV_MAT_ELEM(*mask, uchar, ii, jj) = KNOWN;
			for (q = 0; q<4; q++) {
				if (q == 0) { i = ii - 1; j = jj; }
				else if (q == 1) { i = ii;   j = jj - 1; }
				else if (q == 2) { i = ii + 1; j = jj; }
				else if (q == 3) { i = ii;   j = jj + 1; }
				if ((i <= 1) || (j <= 1) || (i>t->rows - 1) || (j>t->cols - 1)) continue;

				if (CV_MAT_ELEM(*mask, uchar, i, j) == INSIDE) {
					dist = min4(FastMarching_solve(i - 1, j, i, j - 1, mask, t),
						FastMarching_solve(i + 1, j, i, j - 1, mask, t),
						FastMarching_solve(i - 1, j, i, j + 1, mask, t),
						FastMarching_solve(i + 1, j, i, j + 1, mask, t));
					CV_MAT_ELEM(*t, float, i, j) = dist;

					for (color = 0; color <= 0; color++) {
						CvPoint2D32f gradI, gradT, r;
						float Ia = 0, Jx = 0, Jy = 0, s = 1.0e-20f, w, dst, lev, dir, sat;

						if (CV_MAT_ELEM(*mask, uchar, i, j + 1) != INSIDE) {
							if (CV_MAT_ELEM(*mask, uchar, i, j - 1) != INSIDE) {
								gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j + 1) - CV_MAT_ELEM(*t, float, i, j - 1)))*0.5f;
							}
							else {
								gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j + 1) - CV_MAT_ELEM(*t, float, i, j)));
							}
						}
						else {
							if (CV_MAT_ELEM(*mask, uchar, i, j - 1) != INSIDE) {
								gradT.x = (float)((CV_MAT_ELEM(*t, float, i, j) - CV_MAT_ELEM(*t, float, i, j - 1)));
							}
							else {
								gradT.x = 0;
							}
						}
						if (CV_MAT_ELEM(*mask, uchar, i + 1, j) != INSIDE) {
							if (CV_MAT_ELEM(*mask, uchar, i - 1, j) != INSIDE) {
								gradT.y = (float)((CV_MAT_ELEM(*t, float, i + 1, j) - CV_MAT_ELEM(*t, float, i - 1, j)))*0.5f;
							}
							else {
								gradT.y = (float)((CV_MAT_ELEM(*t, float, i + 1, j) - CV_MAT_ELEM(*t, float, i, j)));
							}
						}
						else {
							if (CV_MAT_ELEM(*mask, uchar, i - 1, j) != INSIDE) {
								gradT.y = (float)((CV_MAT_ELEM(*t, float, i, j) - CV_MAT_ELEM(*t, float, i - 1, j)));
							}
							else {
								gradT.y = 0;
							}
						}
						for (k = i - range; k <= i + range; k++) {
							int km = k - 1 + (k == 1), kp = k - 1 - (k == t->rows - 2);
							for (l = j - range; l <= j + range; l++) {
								int lm = l - 1 + (l == 1), lp = l - 1 - (l == t->cols - 2);
								if (k>0 && l>0 && k<t->rows - 1 && l<t->cols - 1) {
									if ((CV_MAT_ELEM(*mask, uchar, k, l) != INSIDE) &&
										((l - j)*(l - j) + (k - i)*(k - i) <= range*range)) {
										r.y = (float)(i - k);
										r.x = (float)(j - l);

										dst = (float)(1. / (VectorLength(r)*sqrt(VectorLength(r))));
										lev = (float)(1. / (1 + fabs(CV_MAT_ELEM(*t, float, k, l) - CV_MAT_ELEM(*t, float, i, j))));

										dir = VectorScalMult(r, gradT);
										if (fabs(dir) <= 0.01) dir = 0.000001f;
										w = (float)fabs(dst*lev*dir);

										if (CV_MAT_ELEM(*mask, uchar, k, l + 1) != INSIDE) {
											if (CV_MAT_ELEM(*mask, uchar, k, l - 1) != INSIDE) {
												gradI.x = (float)((CV_MAT_ELEM(*out, uchar, km, lp + 1) - CV_MAT_ELEM(*out, uchar, km, lm - 1)))*2.0f;
											}
											else {
												gradI.x = (float)((CV_MAT_ELEM(*out, uchar, km, lp + 1) - CV_MAT_ELEM(*out, uchar, km, lm)));
											}
										}
										else {
											if (CV_MAT_ELEM(*mask, uchar, k, l - 1) != INSIDE) {
												gradI.x = (float)((CV_MAT_ELEM(*out, uchar, km, lp) - CV_MAT_ELEM(*out, uchar, km, lm - 1)));
											}
											else {
												gradI.x = 0;
											}
										}
										if (CV_MAT_ELEM(*mask, uchar, k + 1, l) != INSIDE) {
											if (CV_MAT_ELEM(*mask, uchar, k - 1, l) != INSIDE) {
												gradI.y = (float)((CV_MAT_ELEM(*out, uchar, kp + 1, lm) - CV_MAT_ELEM(*out, uchar, km - 1, lm)))*2.0f;
											}
											else {
												gradI.y = (float)((CV_MAT_ELEM(*out, uchar, kp + 1, lm) - CV_MAT_ELEM(*out, uchar, km, lm)));
											}
										}
										else {
											if (CV_MAT_ELEM(*mask, uchar, k - 1, l) != INSIDE) {
												gradI.y = (float)((CV_MAT_ELEM(*out, uchar, kp, lm) - CV_MAT_ELEM(*out, uchar, km - 1, lm)));
											}
											else {
												gradI.y = 0;
											}
										}
										Ia += (float)w * (float)(CV_MAT_ELEM(*out, uchar, km, lm));
										Jx -= (float)w * (float)(gradI.x*r.x);
										Jy -= (float)w * (float)(gradI.y*r.y);
										s += w;
									}
								}
							}
						}
						sat = (float)((Ia / s + (Jx + Jy) / (sqrt(Jx*Jx + Jy*Jy) + 1.0e-20f) + 0.5f));
						{
							CV_MAT_ELEM(*out, uchar, i - 1, j - 1) = cv::saturate_cast<uchar>(sat);
						}
					}

					CV_MAT_ELEM(*mask, uchar, i, j) = BAND;
					Heap->Push(i, j, dist);
				}
			}
		}
	}
}

//注意看0行，0列，erows-1行，cols-1列
#define SET_BORDER1_C1(image,type,value) {\
      int i,j;\
      for(j=0; j<image->cols; j++) {\
         CV_MAT_ELEM(*image,type,0,j) = value;\
	        }\
      for (i=1; i<image->rows-1; i++) {\
         CV_MAT_ELEM(*image,type,i,0) = CV_MAT_ELEM(*image,type,i,image->cols-1) = value;\
	        }\
      for(j=0; j<image->cols; j++) {\
         CV_MAT_ELEM(*image,type,erows-1,j) = value;\
	        }\
   }

//把dst设为待修复（UNKNOWN）的INSIDE 值。 CV_MAT_ELEM 是opencv中用来访问矩阵每个元素的宏，这个宏只对单通道矩阵有效
#define COPY_MASK_BORDER1_C1(src,dst,type) {\
      int i,j;\
      for (i=0; i<src->rows; i++) {\
         for(j=0; j<src->cols; j++) {\
            if (CV_MAT_ELEM(*src,type,i,j)!=0) \
				CV_MAT_ELEM(*dst,type,i+1,j+1) = INSIDE;\
		          }\
	        }\
   }


////////////////////////////////////////////////////////////////////////////////////////
//第二步，修复相关数据初始化处理
//f:flag矩阵   t:T值矩阵
//
void 
mInpaint(const CvArr* _input_img, const CvArr* _inpaint_mask, CvArr* _output_img, double inpaintRange)
{
	cv::Ptr<CvMat> mask, band, flag, t;// mask：标记要修复的地方的掩码矩阵		band：修复的边	 f:flag矩阵   t:T值矩阵   

	cv::Ptr<PriorityQueueFloat> Heap;   

	IplConvKernel* el_cross;  
	el_cross = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
							/*结构元  十字  初始化NarrowBand. 
							 （IplConvKernel定义了一个结构用于描述形态学中的结构元素）
							   其中的变量定义做一简单的描述：

							   nCols，nRows：结构元素的行宽与列高；

							   anchorX，anchorY：结构元素原点（锚点）的位置坐标，水平，垂直；

							   nShiftR：用于表示结构元素的形状类型，有如下几个值：

							   #define  CV_SHAPE_RECT      0

							   #define  CV_SHAPE_CROSS     1    //十字

							   #define  CV_SHAPE_ELLIPSE   2

							   #define  CV_SHAPE_CUSTOM    100

							   分别表示矩形，十字，椭圆和自定义。

							   values：当nShiftR为自定义时，value是指向结构元素数据的指针，如果结构元素的大小定义为8*6，那么values为48长的int数组，值为0或1。
							   */


	CvMat input_hdr, mask_hdr, output_hdr;
	CvMat* input_img, *inpaint_mask, *output_img;

	int range = cvRound(inpaintRange);  //=11,cvRound，函数的一种，对一个double型的数进行四舍五入。
	range = MAX(range, 1);
	range = MIN(range, 100);   //  1<=range<=100,界限划定，不能太小，太大

	//输入图像，掩码矩阵，输出图像 传入
	input_img = cvGetMat(_input_img, &input_hdr);//矩阵input_hdr只是根据_input_img生成一个矩阵头，而其数据指向_input_img的数据。
	inpaint_mask = cvGetMat(_inpaint_mask, &mask_hdr);//矩阵mask_hdr是修复的掩码矩阵，这里只是根据_inpaint_mask生成一个矩阵头，而其数据指向_inpaint_mask的数据。
	output_img = cvGetMat(_output_img, &output_hdr);//矩阵output_hdr只是根据_output_img生成一个矩阵头，而其数据指向_output_img的数据。

	int erows, ecols;   //获取输入待修复图像的行列
	ecols = input_img->cols + 2;//+2是为了修复边缘的时候，不至于没有4邻域
	erows = input_img->rows + 2;
	
	//各类矩阵初始化
	flag.reset(cvCreateMat(erows, ecols, CV_8UC1));    //cvCreateMat创建矩阵，reset初始化 ，CV_8UC1：灰度图
	t.reset(cvCreateMat(erows, ecols, CV_32FC1));		//32位浮点数储存
	band.reset(cvCreateMat(erows, ecols, CV_8UC1));
	mask.reset(cvCreateMat(erows, ecols, CV_8UC1));   

	cvCopy(input_img, output_img);					//拷贝input_img给output_img
	cvSet(mask, cvScalar(KNOWN, 0, 0, 0));		//为mask的每个元素初始化设置数值为KNOWN。cvScalar中只使用第一个通道，因为mask是灰度图

	COPY_MASK_BORDER1_C1(inpaint_mask, mask, uchar);     //根据inpaint_mask给mask赋值inside

	//初始化标记修复的flag矩阵，先全部标记为KNOWN,后面再根据MASK和band标记剩下的类别
	cvSet(flag, cvScalar(KNOWN, 0, 0, 0));
	//标记修复优先度的T值矩阵
	cvSet(t, cvScalar(1.0e6f, 0, 0, 0));   //将T矩阵中元素每个初始化为值10^6

	cvDilate(mask, band, el_cross, 1);				//对mask按照el_cross膨胀1次处理后赋值给band。cvDilate是一个函数，可以用来对输入图像使用指定的结构元进行膨胀。
	Heap = cv::makePtr<PriorityQueueFloat>();

	if (!Heap->Init(band))   //根据band分配链表空间，这里不用减去后的band矩阵是因为这样做不需要多次分配内存空间给节点的压入，弹出等繁琐的工作
		exit;
	cvSub(band, mask, band, NULL);		 //矩阵减法运算,膨胀后的band减去mask得到初始narrowBand存在band中
	SET_BORDER1_C1(band, uchar, 0);		//band的边界设为0
	if (!Heap->Add(band))	//加入narrowBand,得到升序链表NarrowBand，一开始的链表T值都一致，更新后有差别
		exit;

	//初始化flag矩阵  band  inside类
	cvSet(flag, cvScalar(BAND, 0, 0, 0), band);			 
	cvSet(flag, cvScalar(INSIDE, 0, 0, 0), mask);

	cvSet(t, cvScalar(0, 0, 0, 0), band);	 //T矩阵初始化  band为0 其余为10^6

	myTelea_InpaintFMM(mask, t, output_img, range, Heap);//第三步，像素修复程序

	cvReleaseStructuringElement(&el_cross);//释放空间
}


////////////////////////////////////////////////////////////////////////////////////////
//第一步，创建相关副本以处理
//
void imgInpaint(InputArray _src, InputArray _mask, OutputArray _dst,
	double inpaintRange)
{
	Mat src = _src.getMat(), mask = _mask.getMat();//获取原图像和掩码图像，getMat（）函数将传入的参数转换为Mat的结构
	_dst.create(src.size(), src.type());  //初始化了输出图像矩阵，与源图像等尺寸，等类型

	CvMat c_src = src, c_mask = mask, c_dst = _dst.getMat();   //不复制数据，只复制数据的矩阵头。对于数据的修改是直接影响数据本身的

	mInpaint(&c_src, &c_mask, &c_dst, inpaintRange);   //因为是地址信息，要+&
}