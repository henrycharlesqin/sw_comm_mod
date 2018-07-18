#include <stdio.h>
#include "slave.h"
#include "dma.h"
#include "type.h"

extern __thread_local FFT_PARAM slaveParam;
extern __thread_local THREADINFO threadInfo;
extern __thread_local DATAEXCHANGE_INFO* dataInfo;
extern __thread_local FFT_PARAM fft_param;
extern __thread_local int thread_id = 0;

// normal core
void n_recv_row_token()
{
  unsigned short token;
  LONG_GETR(token);
	
	if (token != threadInfo.token)
		threadInfo.token = token;

	// TODO change state

	if (IN_SOME_ROW(threadInfo.range, token))
  {
  	// send temp to token core
  	send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, token);
  }
  else // not in the same row
  {
  	// send temp to comm core
  	send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, threadInfo.rows_comm_core);
  }

  ++threadInfo.current_core;

  if (threadInfo.current_core < threadInfo.cores_in_group)
  {
    // prepare next core data to temp
    data_prepare(dataInfo, &fft_param);
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND;    
  }
  
  // send token to next
  LONG_PUTR(token, threadInfo.next_col_index);
}


// current core
void cu_recv_row_token()
{
  unsigned short token;
  LONG_GETR(token);
	
	if (token != threadInfo.token)
		threadInfo.token = token;

	++threadInfo.current_core;

  if (threadInfo.current_core < threadInfo.cores_in_group)
  {
    // prepare next core data to temp
    data_prepare(dataInfo, &fft_param);
    
    threadInfo.state = RIGHT_RECVRTOKEN;

    LONG_PUTC(threadInfo.current_core, threadInfo.next_col_index);
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND;
  }

	// TODO change state
}

// comm core
void co_recv_row_token()
{
  unsigned short token;
  LONG_GETR(token);

  // TODO change state

	if (IN_SOME_ROW(threadInfo.range, token))
  {
  	// send temp to token core
  	send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, token);
  }
  else // not in the same row
  {
  	// send temp to comm core
  	send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, threadInfo.rows_comm_core);
  }

  
}

void data_prepare(dataexchange_info_t* info, fft_param_t1* param)
{
  unsigned short i0;
  unsigned short i1; 
  unsigned short bufstride = param->bufstride;
  unsigned short is = param->is;
  unsigned short ivs = param->ivs;
  unsigned short index = 0;

  FFT_TYPE *input = info.input_buffer + threadInfo.current_core * 1; // TODO: pay attension to shift value 1.

	if (threadInfo.logic_id == threadInfo.token)
	{
	  // copy data from input buffer to recv_buffer
	  FFT_TYPE *recv = info.recv_buffer;
	  
	  for (i1 = 0; i1 < param->v1; ++i1)
	  {
	    for (i0 = 0; i0 < param->n; ++i0)
	    {
	      index = i0 * is + i1 * ivs;
	    	if (IN_RECV_RANGE(threadInfo.recv_data_range ,index))
	    	{
	    	  // i0 * bufstride + i1 * 1
	        recv[i0 * bufstride + i1 * 1].re = input[index].re; // bufstride 44(20)  is 50(25) ivs 1000(500)
	        recv[i0 * bufstride + i1 * 1].im = input[index].im;
	        ++info.recv_data_len;
	      }
	      else if (OUT_RECV_RANGE(threadInfo.recv_data_range ,index))
	      {
	      	break;
	      }

	      ++info.recv_data_index;
	    }
	  }
	}
	else
	{
	  // copy data from input buffer to tmp buffer
	  FFT_TYPE *recv = info.tmp_buffer;
	  
	  for (i1 = 0; i1 < param->v1; ++i1)
	  {
	    for (i0 = 0; i0 < param->n; ++i0)
	    {
	      index = i0 * is + i1 * ivs;
	    	if (IN_RECV_RANGE(threadInfo.recv_data_range ,index))
	    	{
	    	  // i0 * bufstride + i1 * 1
	        recv[info.tmp_data_index].re = input[index].re; // bufstride 44(20)  is 50(25) ivs 1000(500)
	        recv[info.tmp_data_index].im = input[index].im;
	        ++info.tmp_data_index;
	      }
	      else if (OUT_RECV_RANGE(threadInfo.recv_data_range ,index))
	      {
	      	break;
	      }
	    }
	  }
	}
}

void init_data_exchange()
{
    threadInfo.exchange_info.recv_data_index = 0;
    threadInfo.exchange_info.tmp_data_index = 0;

    // cal mode bat or single
    dataInfo = &threadInfo.exchange_info;
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

void start_data_exchange()
{
  //use threadinfo to 
  threadInfo.token = 0;
  threadInfo.current_core = 0;
  if (IS_BEGIN_CORE(threadInfo.range, threadInfo.logic_id))
  {
    threadInfo.recv_token_time = 1;
    
    // copy data to out
    data_prepare(dataInfo, &fft_param);

    if (!IS_SINGLE_CORE(threadInfo.next_col_index))
    {
      LONG_PUTR(threadInfo.token, threadInfo.next_col_index);
		  threadInfo.state = RIGHT_RECVRDATA;
		}
		else
		{
		  LONG_PUTC(threadInfo.token, threadInfo.next_row_index);
		  threadInfo.state = RIGHT_RECVCDATA;
		}
  }
  else
  {
    threadInfo.recv_token_time = 0;
    
    // copy data to temp
    data_prepare(dataInfo, &fft_param);

    if (!IS_SINGLE_CORE(threadInfo.next_col_index))
      threadInfo.state = RIGHT_RECVRTOKEN;
    else
      threadInfo.state = RIGHT_RECVCTOKEN;
  }
}

inline void end_data_exchange()
{
  threadInfo.token = 0;
}

// normal core recv row token
void normal_recv_row_token()
{
	unsigned short token;
  LONG_GETR(token);
	
	if (token != threadInfo.token)
		threadInfo.token = token;

	if (IN_SOME_ROW(threadInfo.range, token))
  {
  	// send temp to token core
  	send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, token);

  	if (threadInfo.logic_id == threadInfo.rows_comm_core)
  	{
      LONG_PUTC(token, threadInfo.next_row_index);
      threadInfo.state = SUBRIGHT_COMM_RECVCDATA;
  		return;
  	}
  }
  else // not in the same row
  {
  	if (threadInfo.logic_id != threadInfo.rows_comm_core)
  	{
  	   // send temp to comm core
  	   send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, threadInfo.rows_comm_core);
  	}
  	else
  	{
  	   threadInfo.state = SUBRIGHT_COMM_RECVCTOKEN;
  		 return; 
  		 // to next row
  		  
  		 //send_row_data(dataInfo->tmp_buffer, dataInfo->tmp_data_index, threadInfo.rows_comm_core);

  		 //++threadInfo.current_core;

  		 //data_prepare(dataInfo, &fft_param);
  	}
  }

  ++threadInfo.current_core;

  // TODO:end 
  		
  // prepare next core data to temp
  data_prepare(dataInfo, &fft_param);

  // send token to next
  LONG_PUTR(token, threadInfo.next_col_index);

  // end
}

// current core recv row token (token == logic_id)
void current_recv_row_token()
{
	unsigned short token;
  LONG_GETR(token);

  ++threadInfo.recv_token_time;
  if (1 == threadInfo.recv_token_time)
  {
  	// copy data to out
    threadInfo.state = SUBRIGHT_CURRENT_RECVRDATA;

    LONG_PUTC(token, threadInfo.next_col_index);
  }
  else if (2 == threadInfo.recv_token_time)
  {
    threadInfo.recv_token_time = 0;

    if (threadInfo.logic_id == threadInfo.rows_comm_core)
  	{
      LONG_PUTC(token, threadInfo.next_row_index);
      threadInfo.state = SUBRIGHT_COMM_RECVCDATA;
  		return;
  	}
      
    ++threadInfo.current_core;

    if (threadInfo.current_core < threadInfo.cores_in_group)
    {
      threadInfo.state = SUBRIGHT_NORMAL_RECVRTOKEN;

      LONG_PUTC(threadInfo.current_core, threadInfo.next_col_index);
    }
    else
    {
      threadInfo.state = RIGHT_ALLEND;
        
      //LONG_PUTC(token, threadInfo.next_col_index);  last recv token.
    }
  }
}

void current_recv_row_data()
{

}

void comm_recv_row_data()
{
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

void recv_row_data()
{
  int i;
  int length;
  int index;

  // 判断是否为行将通信的core，如果是，通信长度为本行所有数据长
  // 如果不是等待接收网络中所有数据
  if (threadInfo.logic_id != threadInfo.rows_comm_core)
  {
  	length = (GET_ROW_CORES(threadInfo.range) + 1) * 80;
  }
  else
  {
  	length = threadInfo.cores_in_group * 80;
  }

  index = dataInfo->recv_data_index;
  
  for (i = 0; i < length; ++i)
  {
    if (dataInfo->recv_buffer_size <= index)
    {
      index = 0;
    }
    
    LONG_GETR(dataInfo.recv_buffer[index]);

    ++index;
    // 注意循环
  }

  dataInfo->recv_data_index = index;
}

void recv_column_token()
{
}

void recv_column_data()
{
}

// normal core only current state and normal state
void do_normal_data_exchange()
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
			default:
      return;
		}
	}
}

// communicate core
void do_comm_data_exchange()
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
      return;
		}
	}
}

void do_data_exchange()
{
  while (true)
  {
    switch (threadInfo.state)
    {
#if 1
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
#endif
#if 0
      case SUBRIGHT_NORMAL_RECVRTOKEN:
      {
        normal_recv_row_token();
      }
      break;
      case SUBRIGHT_CURRENT_RECVRTOKEN:
      {
        current_recv_row_token();
      }
      break;
      case SUBRIGHT_NORMAL_RECVRDATA:
      break
      case SUBRIGHT_CURRENT_RECVRDATA:
      break;
      case SUBRIGHT_COMM_RECVCDATA:
      break;
      case SUBRIGHT_CURRENT_RECVCDATA:
      break;
      case SUBRIGHT_SINGLE_RECVCDATA:
      break;
      case SUBRIGHT_COMM_RECVCTOKEN:
      break;
      case SUBRIGHT_CURRENT_RECVCTOKEN:
      break;
      case SUBRIGHT_SINGLE_RECVCTOKEN:
      break;
      default:
      break;
#endif
    }
  }
}

