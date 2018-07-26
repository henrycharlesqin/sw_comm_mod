#include <stdio.h>
#include "slave.h"
#include "dma.h"
#include "ldm_malloc.h"
#include "type.h"

extern __thread_local FFT_PARAM slaveParam;
extern __thread_local THREADINFO threadInfo;
extern __thread_local DATAEXCHANGE_INFO dataInfo;
extern __thread_local DATAEXCHANGE_FUNC exchangeFunc;
extern __thread_local FFT_MSG_PARAM fft_msg;
extern __thread_local int thread_id;

void do_core_state_change();

void data_prepare(fft_param_t1* param)
{
  unsigned short i0;
  unsigned short i1; 
  unsigned short bufstride = param->bufstride;
  unsigned short is = param->is;
  unsigned short ivs = param->ivs;
  unsigned short index = 0;

  FFT_TYPE *input = dataInfo.input_buffer + threadInfo.current_core * 1; // TODO: pay attension to shift value 1.

	if (threadInfo.logic_id == threadInfo.token)
	{
	  // copy data from input buffer to recv_buffer
	  FFT_TYPE *recv = dataInfo.recv_buffer;
	  
	  for (i1 = 0; i1 < param->v1; ++i1)
	  {
	    for (i0 = 0; i0 < param->n; ++i0)
	    {
	      index = i0 * is + i1 * ivs;
	    	if (IN_RECV_RANGE(dataInfo.recv_data_range ,index))
	    	{
	    	  // i0 * bufstride + i1 * 1
	        recv[i1 * bufstride + i0 * 1].re = input[index].re; // bufstride 44(20)  is 50(25) ivs 1000(500)
	        recv[i1 * bufstride + i0 * 1].im = input[index].im;
	        ++dataInfo.recv_data_len;
	      }
	      else if (OUT_RECV_RANGE(dataInfo.recv_data_range ,index))
	      {
	      	break;
	      }

	      ++dataInfo.recv_data_index;
	    }
	  }
	}
	else
	{
	  // copy data from input buffer to tmp buffer
	  FFT_TYPE *recv = dataInfo.tmp_buffer;
	  
	  for (i1 = 0; i1 < param->v1; ++i1)
	  {
	    for (i0 = 0; i0 < param->n; ++i0)
	    {
	      index = i0 * is + i1 * ivs;
	    	if (IN_RECV_RANGE(dataInfo.recv_data_range ,index))
	    	{
	    	  // i0 * bufstride + i1 * 1
	        recv[dataInfo.tmp_data_index].re = input[index - START_RECV_INDEX(dataInfo.recv_data_range)].re; // bufstride 44(20)  is 50(25) ivs 1000(500)
	        recv[dataInfo.tmp_data_index].im = input[index - START_RECV_INDEX(dataInfo.recv_data_range)].im;
	        ++dataInfo.tmp_data_index;
	      }
	      else if (OUT_RECV_RANGE(dataInfo.recv_data_range ,index))
	      {
	      	break;
	      }
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
		LONG_PUTR(buffer[i].re, col_index);
		LONG_PUTR(buffer[i].im, col_index);
	}
}

inline unsigned short get_token_col_index(unsigned short token)
{
  int i;
	short dis;
	short col_index;
	
	dis = threadInfo.logic_id - token;


	if (DIR_RIGHT == threadInfo.direction)
	{
		col_index = CORE_COL(threadInfo.physical_id - dis);
	}
	else
	{
		col_index = CORE_COL(threadInfo.physical_id+ dis);
	}	

	return col_index;
}

inline unsigned short get_token_row_index(unsigned short token)
{
	int i;
	short dis = 0;
	short row_index;
	
	unsigned short begin = GET_LOWER_8BITS(threadInfo.range);
	unsigned short end	= GET_HIGHER_8BITS(threadInfo.range);
	
	if (token > end)
	{
		while (end < token)
		{
			++dis;
			end += MAX_CCORE;
		}
	}
	else if (token < begin)
	{
		while (token < begin)
		{
			--dis;
			begin -= MAX_CCORE;
		}
	}
	
	row_index = CORE_ROW(threadInfo.physical_id) + dis;

	return row_index;

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
    LONG_PUTC(buffer[i].re, row_index);
    LONG_PUTC(buffer[i].im, row_index);
  }
  	
}

// normal core
void n_recv_row_token()
{
  unsigned short token;
  LONG_GETR(token);
	
	if (token != threadInfo.token)
		threadInfo.token = token;

	// change state no->cu
	if (token == threadInfo.logic_id)
	{
	  do_core_state_change();
	  return;
	}

	if (IN_SOME_ROW(threadInfo.range, token))
  {
  	// send temp to token core
  	send_row_data(dataInfo.tmp_buffer, dataInfo.tmp_data_index, token);
  }
  else // not in the same row
  {
  	// send temp to comm core
  	send_row_data(dataInfo.tmp_buffer, dataInfo.tmp_data_index, threadInfo.rows_comm_core);
  }

  ++threadInfo.current_core;

  if (threadInfo.current_core < threadInfo.cores_in_group)
  {
    // prepare next core data to temp
    data_prepare(&fft_msg);
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND;    
  }
  
  // send token to next
  if (threadInfo.next_col_index != get_token_col_index(token))
    LONG_PUTR(token, threadInfo.next_col_index);
}

// current core
void cu_recv_row_token()
{
	// ++threadInfo.current_core;

  // prepare next core data to temp
  // data_prepare(&dataInfo, &fft_param);
    
  threadInfo.state = RIGHT_RECVRDATA;

  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);
}

// current core
void cu_recv_row_data()
{
  int i;
  int length;
  int index;

  // get all cores data.
  length = (threadInfo.cores_in_group - 1) * 80;

  index = dataInfo.recv_data_index;
  
  for (i = 0; i < length; ++i)
  {
    if (dataInfo.recv_buffer_size <= index)
    {
      index = 0;
    }
    
    LONG_GETR(dataInfo.recv_buffer[index].re);
    LONG_GETR(dataInfo.recv_buffer[index].im);

    ++index;
    
    // 注意循环
  }

  dataInfo.recv_data_index = index;

  dataInfo.recv_data_len += length;

  ++threadInfo.current_core;

  data_prepare(&fft_msg);

  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

  threadInfo.state = RIGHT_RECVRTOKEN;

  //change core state
	do_core_state_change();

}

// comm core (current core in local row and not comm core itself)
void co_recv_row_token()
{
  unsigned short token;
  LONG_GETR(token);

  if (token != threadInfo.token)
		threadInfo.token = token;

  // change state co->cuco
  if (token == threadInfo.logic_id)
  {
    do_core_state_change();
    return;
  }

  // send temp to token core
  send_row_data(dataInfo.tmp_buffer, dataInfo.tmp_data_index, token);

  LONG_PUTC(token, threadInfo.next_row_index);

  threadInfo.state = RIGHT_RECVCDATA;
}

void co_recv_col_data()
{
	unsigned short token;
	int i,index,length,j;

	token = threadInfo.token;

  // recv all other row data.
  // get all cores data.
  j = threadInfo.cores_in_group - GET_ROW_CORES(threadInfo.range);
  
  length =  80;

  index = 0;

  while (0 < j)
  {
    for (i = 0; i < length; ++i)
    { 
      LONG_GETC(dataInfo.tmp_buffer[index].re);
      LONG_GETC(dataInfo.tmp_buffer[index].im);

      ++index;
    
      // 注意循环
    }
  
    // send tmp data to current core
    send_row_data(dataInfo.tmp_buffer, index, token);

    ++j;

    index = 0;
  }

  ++threadInfo.current_core;

  data_prepare(&fft_msg);
  
  // send token
  LONG_PUTR(threadInfo.token, threadInfo.next_col_index);

  threadInfo.state = RIGHT_RECVRTOKEN;
}

// current and comm core or current core in other row 
// first time recv token, co core -> current core.
void cuco_recv_row_token()
{
	//++threadInfo.current_core;

  //data_prepare(&dataInfo, &fft_param);
    
  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

  threadInfo.state = RIGHT_RECVRDATA;
}

void cuco_recv_row_data()
{
  int i;
  int length;
  int index;

  // get local row all cores data.
  length = (GET_ROW_CORES(threadInfo.range) - 1) * 80;

  if (threadInfo.logic_id == threadInfo.token)
  {
    index = dataInfo.recv_data_index;
  
    for (i = 0; i < length; ++i)
    {
      if (dataInfo.recv_buffer_size <= index)
      {
        index = 0;
      }
    
      LONG_GETR(dataInfo.recv_buffer[index].re);
      LONG_GETR(dataInfo.recv_buffer[index].im);

      ++index;
    
     // 注意循环
    }
    dataInfo.recv_data_index = index;

    dataInfo.recv_data_len += length;

    LONG_PUTC(threadInfo.token, threadInfo.next_row_index); // col token

    threadInfo.state = RIGHT_RECVCDATA;
  }
  else
  {
    index = dataInfo.tmp_data_index;

    for (i = 0; i < length; ++i)
    {
      LONG_GETR(dataInfo.tmp_buffer[index].re);
      LONG_GETR(dataInfo.tmp_buffer[index].im);

      ++index;
    }

    threadInfo.state = RIGHT_RECVCTOKEN;
  }
}

// current core not in local row
void cuco_recv_col_token()
{
  unsigned short token;
	LONG_GETC(token);

	if (threadInfo.token != token)
	  threadInfo.token = token;

	send_column_data(dataInfo.tmp_buffer, dataInfo.tmp_data_index, token);

	++threadInfo.current_core;

  data_prepare(&fft_msg);

  // send token to next core in some row
  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

  // if next row core is not token, send column token.
  if (threadInfo.next_row_index != get_token_row_index(token))
    LONG_PUTC(token, threadInfo.next_col_index);

  // core state change
  if (IN_SOME_ROW(threadInfo.range, threadInfo.current_core))
    do_core_state_change();
}

// only current core is comm core
void cuco_recv_col_data()
{
  unsigned short token;
	int i,index,length,j;
	
	token = threadInfo.token;
	
	// recv all other row data.
	// get all cores data.
	j = threadInfo.cores_in_group - GET_ROW_CORES(threadInfo.range);
		
	length =	80;
	
	index = dataInfo.recv_data_index;
	
	while (0 < j)
	{
		for (i = 0; i < length; ++i)
		{ 
		  if (dataInfo.recv_buffer_size <= index)
      {
        index = 0;
      }
      
			LONG_GETC(dataInfo.recv_buffer[index].re);
			LONG_GETC(dataInfo.recv_buffer[index].im);
	
			++index;			
			// 注意循环
		}
		--j;

		dataInfo.recv_data_len += index;
	}

	++threadInfo.current_core;

  data_prepare(&fft_msg);
		
	// send token
	LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

	threadInfo.state = RIGHT_RECVRTOKEN;

	// change core state change to co
	if (IN_SOME_ROW(threadInfo.range, threadInfo.current_core))
    do_core_state_change();
}



void init_data_exchange()
{
  dataInfo.recv_data_index = 0;
  dataInfo.tmp_data_index = 0;
  threadInfo.token = 0;
  threadInfo.current_core = 0;

  dataInfo.recv_buffer = (FFT_TYPE *)ldm_malloc(MAX_PCORE_DATA*sizeof(FFT_TYPE));
  dataInfo.input_buffer = (FFT_TYPE *)ldm_malloc(MAX_PCORE_DATA*sizeof(FFT_TYPE));
  dataInfo.tmp_buffer = (FFT_TYPE *)ldm_malloc(MAX_PCORE_DATA*sizeof(FFT_TYPE) / 2);

  // cal mode bat or single
}

void start_data_exchange()
{
  //use threadinfo to 
  threadInfo.token = 0;
  threadInfo.current_core = 0;
  
  if (IS_BEGIN_CORE(threadInfo.range, threadInfo.logic_id))
  {
    // copy data to out
    data_prepare(&fft_msg);

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
    // copy data to temp
    data_prepare(&fft_msg);

    if (!IS_SINGLE_CORE(threadInfo.next_col_index))
      threadInfo.state = RIGHT_RECVRTOKEN;
    else
      threadInfo.state = RIGHT_RECVCTOKEN;
  }
}

inline void end_data_exchange()
{
  threadInfo.token = 0;
  threadInfo.current_core = 0;
}

void do_core_state_change()
{
  switch (threadInfo.core_state)
  {
    case CORE_STATE_N:
    {
      threadInfo.core_state = CORE_STATE_CU;
      exchangeFunc.recv_rtoken_func = cu_recv_row_token;
      exchangeFunc.recv_rdata_func = cu_recv_row_data;
      exchangeFunc.recv_ctoken_func = NULL;
      exchangeFunc.recv_cdata_func = NULL;
    }
    break;
    case CORE_STATE_CU:
    {
      threadInfo.core_state = CORE_STATE_N;
      exchangeFunc.recv_rtoken_func = n_recv_row_token;
      exchangeFunc.recv_rdata_func = NULL;
      exchangeFunc.recv_ctoken_func = NULL;
      exchangeFunc.recv_cdata_func = NULL;
    }
    break;
    case CORE_STATE_CO:
    {
      threadInfo.core_state = CORE_STATE_CUCO;
      exchangeFunc.recv_rtoken_func = cuco_recv_row_token;
      exchangeFunc.recv_rdata_func = cuco_recv_row_data;
      exchangeFunc.recv_ctoken_func = cuco_recv_col_token;
      exchangeFunc.recv_cdata_func = cuco_recv_col_data;
    }
    break;
    case  CORE_STATE_CUCO:
    {
      threadInfo.core_state = CORE_STATE_CO;
      exchangeFunc.recv_rtoken_func = co_recv_row_token;
      exchangeFunc.recv_rdata_func = NULL;
      exchangeFunc.recv_ctoken_func = NULL;
      exchangeFunc.recv_cdata_func = co_recv_col_data;
    }
    break;
    default:
    break;
  }
}

void do_data_exchange()
{
  while (1)
  {
    switch (threadInfo.state)
    {
      case RIGHT_RECVRTOKEN:
      {
        if (NULL != exchangeFunc.recv_rtoken_func)
          exchangeFunc.recv_rtoken_func();
      }
      break;
      case RIGHT_RECVRDATA:
      {
        if (NULL != exchangeFunc.recv_rdata_func)
          exchangeFunc.recv_rdata_func();
      }
      break;
      case RIGHT_RECVCTOKEN:
      {
        if (NULL != exchangeFunc.recv_ctoken_func)
          exchangeFunc.recv_ctoken_func();
      }
      break;
      case RIGHT_RECVCDATA:
      {
        if (NULL != exchangeFunc.recv_cdata_func)
          exchangeFunc.recv_cdata_func();
      }
      break;
      default:
      break;
    }
  }
}

