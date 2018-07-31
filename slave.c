#include <stdio.h>
#include "slave.h"
#include "dma.h"

#include "type.h"

__thread_local unsigned short index_reversion_table[MAX_RCORE];
__thread_local THREADINFO threadInfo;
__thread_local DATAEXCHANGE_INFO dataInfo;
__thread_local FFT_PARAM fft_param;
__thread_local FFT_MSG_PARAM fft_msg;
__thread_local DATAEXCHANGE_FUNC exchangeFunc;

__thread_local volatile unsigned long get_reply = 0;
__thread_local volatile unsigned long put_reply = 0;
__thread_local int thread_id = 0;
__thread_local unsigned int pre_rows = 0; // 记录本核之前其他核所读取的行数之和

/*默认该数组id升序排列，如数组未满最后一个放0 */
__thread_local FFT_FUNC n_func_array[NUM_32] = 
{
	//{20, n2fv_20},
	//{32, n2fv_32},
	FFT_FUNC_END
};

__thread_local FFT_FUNC t_func_array[NUM_32] =
{
	//{20, t3fv_20},
	//{25, t3fv_25},
	FFT_FUNC_END
};

extern FFT_PARAM masterParam;
extern FFT_TYPE W[3200];
extern FFT_TYPE *OutputBuf;
extern FFT_TYPE *InputBuf;
extern int Rows;

void input_origin_data()
{
  if (threadInfo.group_id >= 1)
    return;
  
  athread_get(PE_MODE, (void*)(InputBuf + dataInfo.input_data_offset), (void*)dataInfo.input_buffer, dataInfo.input_data_len * sizeof(FFT_TYPE), (void *)&get_reply, 0, 0, 0);
	asm ("memb");
	while (get_reply != 1);
	get_reply = 0;
}

void output_results()
{
  if (threadInfo.group_id >= 1)
    return;
  
  athread_put(PE_MODE, (void*)dataInfo.recv_buffer, (void*)(OutputBuf + dataInfo.input_data_offset), dataInfo.input_data_len * sizeof(FFT_TYPE), (void *)&put_reply, 0, 0);
	//athread_put (PE_MODE, (void*)buf_ldm, (void*)step->output + offset, length, (void *)&put_reply, 0, 0);
	asm ("memb");
	while (put_reply != 1);
	put_reply = 0;
}

void fft_func_test(void* param)
{
  int i,j;
	thread_id = athread_get_id(-1);

	fft_msg.bufstride = 20;
	fft_msg.is = 25;
	fft_msg.ivs = 500;
	fft_msg.n = 20;
	fft_msg.v1 = 20;

	init_threadinfo(10000);


	if (0 != threadInfo.group_id)
	  return;

	//if (0 != thread_id)
	  //return;
#if 1
  //printf("%d", threadInfo.physical_id);
  //printf("%d", threadInfo.group_id);
	//printf("%d", threadInfo.logic_id);
	//printf("%d", threadInfo.direction);
	//printf("%d", threadInfo.cores_in_group);
	//printf("%x", threadInfo.rows_comm_core);
	//printf("%d", threadInfo.rows_in_group);
	//printf("%x", threadInfo.range);
	//printf("%x", threadInfo.next_col_index);
	//printf("%x", threadInfo.next_row_index);
	//printf("%d", threadInfo.core_state);
  //printf("%d", threadInfo.rows_in_group);
  //printf("%d", dataInfo.input_data_range);
#endif

	//
#if 0
	if (15 == thread_id)
	{
	  for (i = 0; i < MAX_RCORE; ++i)
	  {
	    printf("\n");
	    for (j = 0; j < MAX_CCORE; ++j)
	    {
	      printf("%d", threadInfo.core_group_map[i][j]);
	    }
	  }
	  return;
	}
	else
	{
	  return;
	}
#endif

  init_data_exchange();

  input_origin_data();

	start_data_exchange();

	do_data_exchange();

	end_data_exchange();

	output_results();
	
}


