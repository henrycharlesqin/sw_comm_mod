#define FFT_TYPE swfftfComplex
#define FFT_NUM 256 
#define FFT_DIS 64
#define SLAVE_THREADS 64
#define CORE_PER_GRP 64
#define ALL_USE_TIME
#define NN 2
#define BUFFER_SIZE 100
#define CORE_NUM 64
#define KNUM 1024
#define MAX_PCORE_DATA 2000

#define MAX_RCORE 8
#define MAX_CCORE 8


#define NUM_20 20
#define NUM_25 25
#define NUM_32 32
#define NUM_64 64

/* 返回值*/
#define RET_OK 0
#define RET_ERR 1
#define DIV_ABLE

#define STEP_1 0
#define STEP_2 1
#define STEP_3 2

/*函数指针类型*/
#define FUNC_TYPE_N 1
#define FUNC_TYPE_T 2


/* 计算*/
#define WS(stride, i) (stride * i)
#define RE(n) (n / CORE_NUM)
#define MOD(n) (n % CORE_NUM)
#define MM_SHUFFLE(fp3,fp2,fp1,fp0) ((fp3 << 6) | (fp2 << 4) | (fp1 << 2) | fp0)

#define GET_HIGHER_16BITS(data) ((data & 0x0FFFF0000) >> 16)
#define GET_LOWER_16BITS(data) ((data & 0x0FFFF))
#define GET_HIGHER_8BITS(data) ((data & 0x0FF00) >> 8)
#define GET_LOWER_8BITS(data) ((data & 0x0FF))

#define SET_16BITS_PARAM(h,l) ((h & 0x0FF << 8) | (l & 0x0FF))
#define SET_32BITS_PARAM(h,l) ((h & 0x0FFFF << 16) | (l & 0x0FFFF))

#define CORE_ROW(index) (index >> 3)
#define CORE_COL(index) (index & 0x07)
#define IN_SOME_ROW(range, logic_id) ((((range & 0x0FF00) >>8) >= logic_id) && ((range & 0x0FF) <= logic_id))
#define IS_BEGIN_CORE(range, logic_id) ((range & 0x0FF) == logic_id)
#define IS_END_CORE(range, logic_id) (((range & 0x0FF00) >>8) == logic_id)
#define IS_SINGLE_CORE(next_core_index) (next_core_index == 0x0FF)
#define BEGIN_CORE_AROW(range) (range & 0x0FF)
#define END_CORE_AROW(range) ((range & 0x0FF00) >>8)
#define GET_ROW_CORES(range) (((range & 0x0FF00) >>8) - (range & 0x0FF) + 1)

#define IN_RECV_RANGE(recv_data_range, index) ((((recv_data_range & 0x0FFFF0000) >> 16) <= index) && (index <= (recv_data_range & 0x0FFFF)))
#define OUT_RECV_RANGE(recv_data_range, index) (index > (recv_data_range & 0x0FFFF))

/* 寄存器间通信*/
// 行发送
#define LONG_PUTR(var,dest)    \
asm volatile ("putr %0,%1"::"r"(var),"r"(dest):"memory")

// 行接收
#define LONG_GETR(var)   \
asm volatile ("getr %0":"=r"(var)::"memory")

// 列发送
#define LONG_PUTC(var,dest)    \
asm volatile ("putc %0,%1\n"::"r"(var),"r"(dest):"memory")

// 列接收
#define LONG_GETC(var)   \
asm volatile ("getc %0\n":"=r"(var)::"memory")



/* 调试*/
#define DEBUG_FILE_OPEN 1

/*state machine*/
#define RIGHT_RECVRTOKEN 1
#define RIGHT_RECVRDATA 2
#define RIGHT_RECVCTOKEN 3
#define RIGHT_RECVCDATA 4
#define RIGHT_ALLEND 100

#define CORE_STATE_CU    1 // current core (logic_id == token)
#define CORE_STATE_N     2 // normal core
#define CORE_STATE_CO    3 // comm core(can communicate each rows)
#define CORE_STATE_CUCO  4 // current comm core or comm core which has current core in other rows

typedef struct{
	float re;
	float im;
}complex_float_t;

typedef complex_float_t FFT_TYPE;

typedef struct{
	FFT_TYPE* input;           // 主核地址
	FFT_TYPE* output;          // 主核地址
	unsigned int circle_num;   // 循环次数
	unsigned int divisible;    // 循环次数是否可以整除
	unsigned int rs;           // 循环递增步长
	unsigned int circle_max;   // 内循环最大值
	unsigned int ivs;
	unsigned int ovs;
	unsigned int wrs;          // 旋转因子步长
}fft_step_t;

typedef struct
{
  fft_step_t steps1[NN];
  fft_step_t steps2[NN];
	unsigned int num1;         // steps1 步骤
	unsigned int num2;         // steps2 步骤
	unsigned int current;      //当前步骤< num1 + num2
	unsigned int circle_num1;  //
	unsigned int circle_num2;  //
	unsigned int n;            // n点fft
	unsigned int r;            // r行n点fft
}fft_param_t;

typedef fft_step_t FFT_STEP;
typedef fft_param_t FFT_PARAM;

typedef void (*func_ptr)(int ,int ,FFT_TYPE*,FFT_TYPE*); // 注意虽然n函数与t函数定义相同，但是第三个参数代表意义不同

typedef struct
{
	unsigned int id;
	func_ptr func;
}fft_func_t;

typedef fft_func_t FFT_FUNC;
#define FFT_FUNC_END {0,0}

/**/
#define FUNC_ARRAY_SIZE(a) (sizeof(a)/sizeof(FFT_FUNC))

typedef struct
{
  unsigned short n;                // N dot FFT
  unsigned short ovs;    
  unsigned short ivs;
  unsigned short v1;
  unsigned short is;
  unsigned short bufstride;
} fft_param_t1;

typedef fft_param_t1 FFT_MSG_PARAM;

typedef void (*recv_row_token_func)();
typedef void (*recv_row_data_func)();
typedef void (*recv_col_token_func)();
typedef void (*recv_col_data_func)();

typedef struct
{
	recv_row_token_func recv_rtoken_func;
	recv_row_data_func recv_rdata_func;
	recv_col_token_func recv_ctoken_func;
	recv_col_data_func recv_cdata_func;
}dataexchange_func_t;

typedef struct
{
  unsigned short recv_data_index;    // recv buffer index
  unsigned short recv_buffer_size;   // recv buffer size
  unsigned short recv_data_len;      // recv data length per core
  unsigned short recv_data_span;     // recv data interval
  unsigned short recv_total_len;     // recv total
  unsigned short input_buffer_size;  // recv data len
  unsigned short tmp_data_index;     // temp buffer index
  unsigned short tmp_buffer_size;    // temp buffer size
  FFT_TYPE *input_buffer;            // dma read origin data
  FFT_TYPE *current_input_buffer;    // input buffer current address
  FFT_TYPE *recv_buffer;             // recv buffer start address
  FFT_TYPE *current_recv_buffer;     // recv buffer current address
  FFT_TYPE *tmp_buffer;              // temp buffer address
}dataexchange_info_t;

typedef void (*data_prepare_ptr)(dataexchange_info_t*, fft_param_t1*);
typedef void (*data_send_ptr)(dataexchange_info_t*, fft_param_t1*);
typedef void (*data_recv_ptr)(dataexchange_info_t*, fft_param_t1*);

typedef dataexchange_info_t DATAEXCHANGE_INFO;
typedef dataexchange_func_t DATAEXCHANGE_FUNC;

#define DIR_RIGHT 1
#define DIR_LEFT  2

typedef struct
{
	unsigned short is_init;         // 1 init.
	unsigned short group_id;        // group id(one core can restore 2000B data, so when N is bigger than 2000B. we divided N by multi-slave cores. this multi-slave cores are called a group)
	unsigned short logic_id;        // when we use full core-array logic_id equal physical_id, but in common we divited cores into many groups, so logic_id is id in a group.
	unsigned short physical_id;		  // thread_id
	unsigned short core_rc_index;   // higher 8 bits column index lower 8 bits row index
	unsigned short next_col_index;  // transfer token to next core higher 8 bits column index lower 8 bits row index(only column index  or row index valid, invalid index is 0)
	unsigned short next_row_index;
	unsigned short rows_comm_core;  // group contains two rows, this core show through which core to communicate in different rows
	unsigned short current_core;    // Other core send data to current core
	unsigned short direction;       // show cores communicate direction.
	unsigned short range;           // show cores logic id from a to b in same row of a group. lower 8 bits begin, higher 8 bits end.
	unsigned short token;
	unsigned short core_state;      // core status.
	unsigned int  recv_data_rem;    // recieve data remainder
	unsigned int  recv_data_range;  // high 16 bit start low 16 bit end
	unsigned char cores_in_group;   // cores in a group
	unsigned char rows_in_group;    // group contains the number of core rows.
	unsigned char current_row;      // row number in group.
	unsigned char state;
	unsigned char core_group_map[MAX_RCORE][MAX_CCORE];  // current group is including which slave cores .	If not included , value is 0x0FF, otherwise value is logic_id.	
}threadinfo_t;

typedef threadinfo_t THREADINFO;

