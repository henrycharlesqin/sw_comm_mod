#include <stdio.h>
#include "slave.h"
#include "dma.h"

#include "type.h"

__thread_local unsigned short index_reversion_table[MAX_RCORE];
__thread_local FFT_TYPE buf_ldm[2*KNUM] __attribute__((__aligned__(128)));
__thread_local FFT_TYPE out_ldm[2*KNUM] __attribute__((__aligned__(128)));
__thread_local FFT_TYPE tmp_ldm[2*NUM_25]  __attribute__((__aligned__(128)));
__thread_local FFT_TYPE w_ldm[KNUM] __attribute__((__aligned__(128)));

__thread_local FFT_PARAM slaveParam;
__thread_local THREADINFO threadInfo;
__thread_local volatile unsigned long get_reply, put_reply;
__thread_local int thread_id = 0;
__thread_local unsigned int pre_rows = 0; // ��¼����֮ǰ����������ȡ������֮��
__thread_local FFT_TYPE W25[] __attribute__((__aligned__(16))) =
{
	{1.0000000,0.0000000},{0.9999692,0.0078539},{1.0000000,0.0000000},{0.9997224,0.0235598},{1.0000000,0.0000000},{0.9975028,0.0706270},{1.0000000,0.0000000},{0.9822873,0.1873813},
{0.9998766,0.0157073},{0.9997224,0.0235598},{0.9988899,0.0471065},{0.9975028,0.0706270},{0.9900237,0.1409012},{0.9775999,0.2104718},{0.9297765,0.3681246},{0.8443279,0.5358268},
{0.9995066,0.0314108},{0.9992290,0.0392598},{0.9955620,0.0941083},{0.9930685,0.1175374},{0.9602937,0.2789911},{0.9381913,0.3461171},{0.7289686,0.6845471},{0.5877853,0.8090170},
{0.9988899,0.0471065},{0.9984891,0.0549502},{0.9900237,0.1409012},{0.9864293,0.1641868},{0.9114033,0.4115144},{0.8800633,0.4748564},{0.4257793,0.9048271},{0.2486899,0.9685832},
{0.9980267,0.0627905},{0.9975028,0.0706270},{0.9822873,0.1873813},{0.9775999,0.2104718},{0.8443279,0.5358268},{0.8043756,0.5941211},{0.0627905,0.9980267},{-0.1253332,0.9921147},
{0.9969173,0.0784591},{0.9962704,0.0862864},{0.9723699,0.2334454},{0.9666001,0.2562894},{0.7604060,0.6494480},{0.7126385,0.7015314},{-0.3090170,0.9510565},{-0.4817537,0.8763067},
{0.9955620,0.0941083},{0.9947921,0.1019245},{0.9602937,0.2789911},{0.9534542,0.3015380},{0.6613119,0.7501111},{0.6066824,0.7949444},{-0.6374240,0.7705132},{-0.7705132,0.6374240},
{0.9939610,0.1097343},{0.9930685,0.1175374},{0.9460854,0.3239174},{0.9381913,0.3461171},{0.5490228,0.8358074},{0.4886212,0.8724960},{-0.8763067,0.4817537},{-0.9510565,0.3090170},
{0.9921147,0.1253332},{0.9910997,0.1331213},{0.9297765,0.3681246},{0.9208455,0.3899277},{0.4257793,0.9048271},{0.3608108,0.9326390},{-0.9921147,0.1253332},{-0.9980267,-0.0627905},
{0.9900237,0.1409012},{0.9888865,0.1486724},{0.9114033,0.4115144},{0.9014551,0.4328726},{0.2940403,0.9557930},{0.2258013,0.9741734},{-0.9685832,-0.2486899},{-0.9048271,-0.4257793},
{0.9876883,0.1564345},{0.9864293,0.1641868},{0.8910065,0.4539905},{0.8800633,0.4748564},{0.1564345,0.9876883},{0.0862864,0.9962704},{-0.8090170,-0.5877853},{-0.6845471,-0.7289686},
{0.9851093,0.1719291},{0.9837286,0.1796607},{0.8686315,0.4954587},{0.8567175,0.5157859},{0.0157073,0.9998766},{-0.0549502,0.9984891},{-0.5358268,-0.8443279},{-0.3681246,-0.9297765},
{0.9822873,0.1873813},{0.9807853,0.1950903},{0.8443279,0.5358268},{0.8314696,0.5555702},{-0.1253332,0.9921147},{-0.1950903,0.9807853},{-0.1873813,-0.9822873},{0.0000000,-1.0000000},
{0.9792228,0.2027873},{0.9775999,0.2104718},{0.8181497,0.5750053},{0.8043756,0.5941211},{-0.2638730,0.9645574},{-0.3313379,0.9435122},{0.1873813,-0.9822873},{0.3681246,-0.9297765},
{0.9759168,0.2181432},{0.9741734,0.2258013},{0.7901550,0.6129071},{0.7754957,0.6313528},{-0.3971479,0.9177546},{-0.4609744,0.8874134},{0.5358268,-0.8443279},{0.6845471,-0.7289686},
{0.9723699,0.2334454},{0.9705065,0.2410751},{0.7604060,0.6494480},{0.7448941,0.6671828},{-0.5224986,0.8526402},{-0.5814132,0.8136084},{0.8090170,-0.5877853},{0.9048271,-0.4257793},
};

/*Ĭ�ϸ�����id�������У�������δ�����һ����0 */
FFT_FUNC n_func_array[NUM_32] = 
{
	{20, n2fv_20},
	{32, n2fv_32},
	FFT_FUNC_END
};

FFT_FUNC t_func_array[NUM_32] =
{
	{20, t3fv_20},
	{25, t3fv_25},
	FFT_FUNC_END
};

extern FFT_PARAM masterParam;
extern FFT_TYPE W[3200];

/*
fft �������߼�
*/
void fft_solve(FFT_STEP* step, int row)
{
	int offset;
    int i = 0;
    int j = 0;
	
	if (NULL == step)
		return;
	
	if ((NUM_20 == step->circle_max) && (STEP_1 == slaveParam.current))
	{
		// nxfvx
		n2fv_20(0, 2 , buf_ldm, out_ldm);
	}
	else if ((NUM_20 == step->circle_max) && (STEP_2 == slaveParam.current))
	{
		offset = thread_id * step->wrs;
		t3fv_20(0, 2, &w_ldm[offset], buf_ldm);
	}
	else if ((NUM_25 == step->circle_max) && ((STEP_2 == slaveParam.current) || (STEP_3 == slaveParam.current)))
	{     
		if (1 == row)
		{
			offset = pre_rows * step->wrs;
			t3fv_25(0, 2, &W25[offset], buf_ldm);
		}
		else
		{
			for (i = 0; i < row; ++i)
			{ 
				//offset = (pre_rows + i) * step->wrs;
				offset = i * step->wrs;
			    // �ò��ִ�������t3fv_25������֧�ֶ��д������뵥�д����������simd��ȡ�������
                for (j = 0; j < NUM_25*2; ++j)
                {
                    tmp_ldm[j].re = buf_ldm[j + i*NUM_25*2].re;
                    tmp_ldm[j].im = buf_ldm[j + i*NUM_25*2].im;
                }  
                t3fv_25(0, 2, &w_ldm[offset], tmp_ldm);
                for (j = 0; j < NUM_25*2; ++j)
                {
                    buf_ldm[j + i*NUM_25*2].re = tmp_ldm[j].re;
                    buf_ldm[j + i*NUM_25*2].im = tmp_ldm[j].im;
                }
            }
		}
	}
	else if ((NUM_32 == step->circle_max) && (STEP_1 == slaveParam.current))
	{
		n2fv_32(0, 2, buf_ldm, out_ldm);
	}
}

/*
�����ȡ������,ͨ��ָ�뷵��֮ǰ����ƫ��ֵ
*/
inline int cal_work_row(unsigned int cn, unsigned int* pre_total)
{
	int re;
	int mod;
	int rows = 0;

	if ((0 == cn) || (NULL == pre_total))
		return rows;

	re = RE(cn); // ����
	if (0 == re)
	{
		if (thread_id < cn)
		{
			*pre_total = thread_id;
			rows = 1;
		}
	}
	else
	{
		mod = MOD(cn); // ������
		if (thread_id < mod)
		{
			rows = re + 1;
			*pre_total = thread_id * rows;
		}
		else // mod Ϊ0 ok
		{
			rows = re;
			*pre_total = mod * (rows + 1) + (thread_id - mod) * rows;
		}
	}

	return rows;
}

/*
�����ʼ������
*/
void fft_func_init(void* param)
{
	unsigned int current;
	FFT_STEP* step;
	int srow_length;
	int arow_length;
  int total_length;
	int circle_num;
	int offset;
	int row;
	int mod;
	
	thread_id = athread_get_id(-1);
	
	// ͨ��DMA ��ȫ��masterParam��ȡ����
	athread_get (PE_MODE, (void*)&masterParam, (void*)&slaveParam, sizeof(FFT_PARAM), (void *)&get_reply, 0, 0, 0);
	asm ("memb");
	while (get_reply != 1);
	get_reply = 0;

	// ���㵱ǰstep index,�ò���Ĭ��Ϊ0����Ϊinit����
	current = slaveParam.current;
	step = &slaveParam.steps1[0];
	mod = step->circle_num % 2;

	// ����circle_num ���ܱ�����������
	if (0 != mod)
	{
		circle_num = step->circle_num / 2 + 1;
	}
	else
	{
		circle_num = step->circle_num / 2;
	}

	// �жϱ������Ƿ�����
	row = cal_work_row(circle_num, &pre_rows);
	if (0 == row)
		return;

	// TODO ��Ҫ�ж����ݴ�С�Ƿ񳬹�2K

	// �����ȡ�����ݣ���Ϊʹ��simd�ʸ�����Ҫ����2
	srow_length = sizeof(FFT_TYPE) * step->circle_max;
	arow_length = srow_length * 2;
  total_length = arow_length * row;
	offset = pre_rows * arow_length;
	athread_get (PE_MODE, (void *)step->input + offset, (void*)&buf_ldm[0], total_length, (void *)&get_reply, 0, 0, 0);
	asm ("memb");
	while (get_reply != 1);
	get_reply = 0;

	// FFT��ϼ���
	fft_solve(step, row);

	// ���ݻش�
	if (0 != mod)
	{
		//���һ�������ٴ���һ������
		if (circle_num == (pre_rows + row))
		{
			total_length -= srow_length;
		}
	}
	athread_put (PE_MODE, (void*)out_ldm, (void*)step->output + offset, total_length, (void *)&put_reply, 0, 0);
	// athread_put (PE_MODE, (void*)buf_ldm, (void*)step->output + offset, length, (void *)&put_reply, 0, 0);
	asm ("memb");
	while (put_reply != 1);
	put_reply = 0;
	
}

void fft_func_proc(void* param)
{
	//unsigned int current = *((unsigned int *)(param));
	unsigned int current;
	FFT_STEP* step;
  int total_length;
	int circle_num;
	int arow_length;
	int srow_length;
	int offset;
	int row;
	int mod;
	void* out_buf = NULL;

	// ͨ��DMA ��ȫ��masterParam��ȡ����
	athread_get (PE_MODE, (void*)&masterParam, (void*)&slaveParam, sizeof(FFT_PARAM), (void *)&get_reply, 0, 0, 0);
	asm ("memb");
	while (get_reply != 1);
	get_reply = 0;
	
	// ���㵱ǰstep index,�ò���Ĭ��Ϊ0����Ϊinit����
	current = slaveParam.current;

    if ((STEP_1 == current) || (STEP_2 == current))
    {
		step = &slaveParam.steps1[current];
    }
	else if (STEP_3 == current)
	{
		step = &slaveParam.steps2[0];
	}
	else
	{
		return;
	}

	// ����circle_num ���ܱ�����������
	mod = step->circle_num % 2;
	if (0 != mod)
	{
		circle_num = step->circle_num / 2 + 1;
	}
	else
	{
		circle_num = step->circle_num / 2;
	}

	// �жϱ������Ƿ�����
	row = cal_work_row(circle_num, &pre_rows);
    if (0 == row)
	{
		return;
	}

	// TODO ��Ҫ�ж����ݴ�С�Ƿ񳬹�2K
	// ͨ��DMA ��ȡ��ת����
	if (STEP_3 == current)
	{
	    athread_get (PE_MODE, (void*)&W[pre_rows*step->wrs], (void*)&w_ldm, step->wrs*row*sizeof(FFT_TYPE), (void *)&get_reply, 0, 0, 0);
        asm ("memb");
	    while (get_reply != 1);
	    get_reply = 0;
	}


	// �����ȡ�����ݣ���Ϊʹ��simd�ʸ�����Ҫ����2
	srow_length = sizeof(FFT_TYPE) * step->circle_max;
	arow_length = srow_length * 2;
    total_length = arow_length * row;
	offset = pre_rows * arow_length;
	athread_get (PE_MODE, (void *)step->input + offset, (void*)&buf_ldm[0], total_length, (void *)&get_reply, 0, 0, 0);
	asm ("memb");
	while (get_reply != 1);
	get_reply = 0;

	// FFT��ϼ���
	fft_solve(step, row);

	// ���ݻش�
	if ((STEP_2 == current) || (STEP_3 == current))
	{
		out_buf = (void*)buf_ldm; // t3**����ʱ���������ʹ������ͬ�Ļ���
	}
	else
	{
		out_buf = (void*)out_ldm;
	}
	
	if (0 != mod)
	{
		//���һ�������ٴ���һ������
		if (circle_num == (pre_rows + row))
		{
			total_length -= srow_length;
		}
	}
	
	athread_put (PE_MODE, (void*)out_buf, (void*)step->output + offset, total_length, (void *)&put_reply, 0, 0);
	//athread_put (PE_MODE, (void*)buf_ldm, (void*)step->output + offset, length, (void *)&put_reply, 0, 0);
	asm ("memb");
	while (put_reply != 1);
	put_reply = 0;
	
}

void fft_func_test(void* param)
{
	//
}

void call_fft_func(int type, int id)
{
	int i = 0;
	switch (type)
	{
		case FUNC_TYPE_N:
			for (; i < FUNC_ARRAY_SIZE(n_func_array); ++i)
			{
				FFT_FUNC* p = n_func_array[i];
				if (NULL == p)
					break;

				if (id < p->id) // Ĭ��������������
				{
					break;
				}
				
				if (id == p->id)
				{
					if (NULL != p->func())
					{
					    p->func();
					    break;
					}
				}
			}
			break;
		case FUNC_TYPE_T:
			for (; i < FUNC_ARRAY_SIZE(t_func_array); ++i)
			{
				FFT_FUNC* p = t_func_array[i];
				if (NULL == p)
					break;

				if (id < p->id) // Ĭ��������������
				{
					break;
				}

				if (id == p->id)				
				{
					p->func();
					break;
				}
			}
			break;
		default:
			break;
				
	}

	return;
}
void init_data_exchange()
{

}

void start_data_exchange()
{
  //use threadinfo to 
  threadInfo.token = 0;
  threadInfo.current_core = 0;
  if (IS_BEGIN_CORE(threadInfo.range, threadInfo.logic_id))
  {
    threadInfo.recv_token_time = 1;
    
    // copy data to out
    LONG_PUTR(threadInfo.token, threadInfo.next_core_index);
		threadInfo.state = RIGHT_RECVRDATA;
  }
  else
  {
    threadInfo.recv_token_time = 0;
    
    // copy data to temp
    threadInfo.state = RIGHT_RECVRTOKEN;
  }
}

inline void end_data_exchange()
{
}

void recv_row_token()
{
  unsigned short token;
  LONG_GETR(token);

  if (token != threadInfo.token)
    threadInfo.token = token;

  // the token is current core?
  if (threadInfo.logic_id != token)
  {
  	if (IN_SOME_ROW(threadInfo.range, token))
  	{
  		// send temp to token core
  		send_row_data(buffer, length, token);

  		++threadInfo.current_core;
  		// prepare next core data to temp

  		// end
  	}
  	else
  	{

  		if (threadInfo.logic_id != threadInfo.rows_comm_core)
  		{
  	    // send temp to comm core
  	    send_row_data(buffer, length, threadInfo.rows_comm_core);

  	    ++threadInfo.current_core;
  		  // prepare next core data to temp

  		  //end
  		}
  		else
  		{
  		  // to next row
  		}
  	}
  }
  else
  {
    ++threadInfo.recv_token_time;
    if (1 == threadInfo.recv_token_time)
    {
  	  // copy data to out
      threadInfo.state = RIGHT_RECVCDATA;

      LONG_PUTC(token, threadInfo.next_core_index);
    }
    else if (2 == threadInfo.recv_token_time)
    {
      threadInfo.recv_token_time = 0;
      
      ++threadInfo.current_core;

      if (threadInfo.current_core < threadInfo.cores_in_group)
      {

        threadInfo.state = RIGHT_RECVCTOKEN;

        LONG_PUTC(threadInfo.current_core, threadInfo.next_core_index);
      }
      else
      {
        threadInfo.state = RIGHT_ALLEND;
        
        LONG_PUTC(token, threadInfo.next_core_index);
      }
    }
  }
}


void send_row_data(FFT_TYPE* buffer, unsigned short length, unsigned short des)
{
  int i;
	short dis;
	short col_index;
	
	dis = threadInfo.logic_id - des;


	if (DIR_RIGHT == threadInfo.direction)
	{
		col_index = CORE_COL(threadInfo.physical_id) - dis;
	}
	else
	{
		col_index = CORE_COL(threadInfo.physical_id) + dis;
	}	

	for (i = 0; i < length; ++i)
	{
		LONG_PUTR(buffer[i], col_index);
	}
}

void send_column_data(FFT_TYPE* buffer, unsigned short length, unsigned short des)
{
  int i;
  short dis = 0;
  short row_index;

	unsigned short begin = GET_LOWER_8BITS(threadInfo.range);
  unsigned short end  = GET_HIGHER_8BITS(threadInfo.range);

  if (des > end)
  {
  	while (end < des)
  	{
  		++dis;
  		end += MAX_CCORE;
  	}
  }
  else if (des < begin)
  {
  	while (des < begin)
  	{
  	  --dis;
  	  begin -= MAX_CCORE;
  	}
  }

  row_index = CORE_ROW(threadInfo.physical_id) + dis;

  for (i = 0; i < length; ++i)
  {
    LONG_PUTC(buffer[i], row_index);
  }
  	
}

void recv_row_data(FFT_TYPE* buffer, unsigned short length, unsigned short start_index)
{
  int i;

  for (i = 0; i < length; ++i)
  {
    LONG_GETR(buffer[start_index]);
  }

}

void recv_column_token()
{
}

void recv_column_data()
{
}

void do_data_exchange()
{
  while (true)
  {
    switch (threadInfo.state)
    {
      case RIGHT_RECVRTOKEN:
      {
        recv_row_token();
      }
      break;
      case RIGHT_RECVRDATA:
      {
        recv_row_data();
      }
      break;
      case RIGHT_RECVCTOKEN:
      {
        recv_column_token();
      }
      break;
      case RIGHT_RECVCDATA:
      {
        recv_column_data();
      }
      break;
      default:
      break;
    }
  }
}

// util.c
unsigned short init_threadinfo(int thread_id)
{
	// cal multi-core read N/n
	int mod = 0;
	int mod1 = 0;
	int quo = 0;
	int cores_per_group = 0;
	int i = 0,j = 0;

	unsigned char begin,end;

	threadInfo.logic_id = thread_id;

	mod = N % MAX_PCORE_DATA;
	quo = N / MAX_PCORE_DATA;
	if (0 == mod)
	{
	  cores_per_group = quo;
	  threadInfo.exchange_info.input_buffer_size = N / cores_per_group;
	}
	else
	{
		cores_per_group = quo + 1;
		mod1 = N % cores_per_group;
		if ( 0 == threadInfo.logic_id)
	  	threadInfo.exchange_info.input_buffer_size = N / cores_per_group + mod1;
	  else
	    threadInfo.exchange_info.input_buffer_size = N / cores_per_group;
	}

	// group id
	threadInfo.physical_id = thread_id;
	threadInfo.group_id = thread_id / cores_per_group;
	threadInfo.cores_in_group = cores_per_group;

	// group map
	init_group_map(threadInfo.group_id, cores_per_group);

	// range
	init_row_range();
	
	// rows_comm_index
	
	return RET_OK
}

/*     after init map like this: when cores_per_group = 5

        slave core array: 0 1 2 3 4 means logic_id

        0 1 2 3 4 0 1 2
        0 4 3 2 1 0 4 3
        1 2 3 4 ......

        0 1 2 3 4               group 0
                     0 1 2
                        4 3       group 1
          4 3 2 1 0             group 2
        0
        1 2 3 4                  group 3
*/
void init_group_map(unsigned short gid, unsigned short cores)
{
	unsigned short g = 0;
	unsigned short c = 0;
	signed short i = 0;
	signed short j = 0;
	signed short b1 = 0;
	signed short e1 = MAX_CCORE;
	signed short b2 = MAX_CCORE - 1;
	signed short e2 = 0;
	unsigned short group_id = gid;
	unsigned short cores_per_group = cores; //13
	signed short dir = DIR_RIGHT; // right mean i from 0 to 7. left means i form 7 to 0.

	// init table.
	for (i = 0; i < MAX_RCORE; ++i)
	{
		for (j = 0; j < MAX_CCORE; ++j)
		{
			threadInfo.core_group_map[i][j] = 0xFF;
		}
	}

	i = 0;
	j = 0;
	while (g <= group_id)
	{
		c = cores_per_group;
		for (; i < MAX_RCORE; ++i)
		{
			if (DIR_RIGHT == dir)
			{
				for (j = b1; j < e1; ++j)
				{
					//map[i][j] = g;
					if (g == group_id)
						threadInfo.core_group_map[i][j] = cores_per_group - c;
						
					--c;

					if (0 == c)
					{
						b1 = j + 1;
						break;
					}
				}
				
				if (0 != c)
				{
					if (cores_per_group < (c + 8))
					{
						dir = DIR_LEFT;
						b2 = MAX_RCORE - 1;
						e2 = 0;
					}
				}
			}
			else if (DIR_LEFT == dir)
			{
				for (j = b2; j >= e2; --j)
				{
					//map[i][j] = g;
					if (g == group_id)
						threadInfo.core_group_map[i][j] = cores_per_group - c;
						
					--c;

					if (0 == c)
					{
						b2 = j - 1;
						break;
					}
				}

				if (0 != c)
				{
					if (cores_per_group < (c + 8))
					{
						dir = DIR_RIGHT;
						b1 = 0;
						e1 = MAX_RCORE;
					}
				}
			}

			if (0 == c)
				break;
		}

		++g;
	}

	// we get group_id array. then we need allocate logic_id to map

	return;
}

void init_row_range()
{
  	int row_index = 0;
  	int col_index = 0;
  	int i = 0;

  	unsigned char begin,end;
  	
	  row_index = CORE_ROW(threadInfo.physical_id);
	  col_index = CORE_COL(threadInfo.physical_id);
	  
		i = 0;
		while (i < MAX_CCORE)
		{
			if (0xFF != threadInfo.core_group_map[row_index][i])
			{
				begin = threadInfo.core_group_map[row_index][i];
				break;
			}
	
			++i;
		}
	
		i = MAX_CCORE - 1;
		while (0 <= i)
		{
			if (0xFF != threadInfo.core_group_map[row_index][i])
			{
				end = threadInfo.core_group_map[row_index][i];
				break;
			}
	
			--i;
		}

		// range
		threadInfo.range = SET_16BITS_PARAM(end, begin);

    // direction
		if (begin < end)
		{
		  threadInfo.direction = DIR_RIGHT;
		}
		else if (begin > end)
		{
		  threadInfo.direction = DIR_LEFT;
		}
		else
		{
			threadInfo.direction = ((i == (MAX_CCORE - 1)) ? DIR_LEFT : DIR_RIGHT);
		}

		// logic_id
		threadInfo.logic_id = threadInfo.core_group_map[row_index][col_index];

		// next core

    // comm core
    threadInfo.rows_in_group = 0;
    threadInfo.rows_comm_core = 0x0FF;
		for (i = 0; i < MAX_RCORE; ++i)
		{
		  if ((0x0FF == threadInfo.core_group_map[i][0]) && (0x0FF == threadInfo.core_group_map[i][MAX_CCORE - 1]))
		    continue;

      if (0x0FF == threadInfo.rows_comm_core)
      {
        if (0x0FF == threadInfo.core_group_map[i][0])
        {
          threadInfo.rows_comm_core = threadInfo.core_group_map[i][MAX_CCORE - 1];
        }
        else if (0x0FF == threadInfo.core_group_map[i][MAX_CCORE - 1])
        {
          threadInfo.rows_comm_core = threadInfo.core_group_map[i][0];
        }
      }
		   
		   ++threadInfo.rows_in_group;
		}
}


