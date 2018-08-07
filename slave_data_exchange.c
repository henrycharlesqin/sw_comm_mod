#include <stdio.h>
#include "slave.h"
#include "dma.h"
#include "ldm_malloc.h"
#include "simd.h"
#include "type.h"

extern __thread_local FFT_PARAM slaveParam;
extern __thread_local THREADINFO threadInfo;
extern __thread_local DATAEXCHANGE_INFO dataInfo;
extern __thread_local DATAEXCHANGE_FUNC exchangeFunc;
extern __thread_local FFT_MSG_PARAM fft_msg;
extern __thread_local int thread_id;

void init_core_state();
void do_core_state_change();

void data_prepare(fft_param_t1* param)
{
  unsigned short i0;
  unsigned short i1; 
  unsigned short bufstride = param->bufstride;
  unsigned short is = param->is;
  unsigned short ivs = param->ivs;
  unsigned short index = 0;
  unsigned short index1 = 0;
  unsigned short offset = 0;
  unsigned short start_index = threadInfo.logic_id * 4;
  unsigned short i = 0;
 
  FFT_TYPE *input = dataInfo.input_buffer + threadInfo.current_core * 5; // TODO: pay attension to shift value 1.
  offset = START_RECV_INDEX(dataInfo.input_data_range);

	if (threadInfo.logic_id == threadInfo.current_core)
	{
	  // copy data from input buffer to recv_buffer
	  FFT_TYPE *recv = dataInfo.recv_buffer;
	  for (i = 0; i < 5; ++i)
	  {
	    for (i1 = start_index; i1 < param->v1; ++i1)
	    {
	      for (i0 = 0; i0 < param->n; ++i0)
	      {
	        index = i0 * is + i1 * ivs;
	    	  if (IN_RECV_RANGE(dataInfo.input_data_range, index))
	    	  {
	    	    // i0 * bufstride + i1 * 1
	    	    index1 = i1 * bufstride + i0 * 1;
	          recv[index1].re = input[index - offset].re; // bufstride 44(20)  is 50(25) ivs 1000(500)
	          recv[index1].im = input[index - offset].im;
	          ++dataInfo.recv_data_len;
	        }
	        else if (OUT_RECV_RANGE(dataInfo.input_data_range, index))
	        {
	      	  break;
	        }
	      }
	    }
	    recv += 400;
	    ++input;
	  }

    dataInfo.recv_data_index = (threadInfo.current_core + 1) * 80;
	}
	else
	{
	  // copy data from input buffer to tmp buffer
	  FFT_TYPE *recv = dataInfo.tmp_buffer;
	  dataInfo.tmp_data_index = 0;
	  for (i = 0; i < 5; ++i)
	  {
	    for (i1 = start_index; i1 < param->v1; ++i1)
	    {
	      for (i0 = 0; i0 < param->n; ++i0)
	      {
	        index = i0 * is + i1 * ivs;
	    	  if (IN_RECV_RANGE(dataInfo.input_data_range, index))
	    	  {
	    	    // i0 * bufstride + i1 * 1
	          recv[dataInfo.tmp_data_index].re = input[index - offset].re; // bufstride 44(20)  is 50(25) ivs 1000(500)
	          recv[dataInfo.tmp_data_index].im = input[index - offset].im;
	          ++dataInfo.tmp_data_index;
	        }
	        else if (OUT_RECV_RANGE(dataInfo.input_data_range ,index))
	        {
	      	  break;
	        }
	      }
	    }
	    ++input;
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

	for (i = 0; i < length; i = i + 2)
	{
		LONG_PUTR(simd_set_floatv4(buffer[i].re,buffer[i].im,buffer[i+1].re,buffer[i+1].im), col_index);
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

  for (i = 0; i < length; i = i + 2)
  {
    LONG_PUTC(simd_set_floatv4(buffer[i].re,buffer[i].im,buffer[i+1].re,buffer[i+1].im), row_index);
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

	if (IN_SAME_ROW(threadInfo.range, token))
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
  if (IN_SAME_ROW(threadInfo.range, token))
  {
    if (threadInfo.next_col_index != get_token_col_index(token))
      LONG_PUTR(token, threadInfo.next_col_index);
  }
  else
  {
    if (threadInfo.next_col_index != get_token_col_index(threadInfo.rows_comm_core))
      LONG_PUTR(threadInfo.token, threadInfo.next_col_index);
  }
  
  
}

// current core
void cu_recv_row_token()
{
  threadInfo.state = RIGHT_RECVRDATA;

  // send token to next
  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);
}

// current core
void cu_recv_row_data()
{
  int i,j,z,k;
  int length;
  int index;
  int index1;
  floatv4 data;
  FFT_TYPE* recv_buffer = dataInfo.recv_buffer;
  
  length = 80;
  // get all cores data.

  // TODO one row two rows  three row procedure is different
  for (j = 0; j < threadInfo.cores_in_group - 1; ++j)
  {
    recv_buffer = dataInfo.recv_buffer;
    index1 = (int)dataInfo.recv_core_seq[j];
    for (z = 0; z < 5 ; ++z)
    {
      index = index1 * 80;
      for (i = 0; i < length / 2; ++i)
      {
        LONG_GETR(data);
        simd_store(data, (float*)&recv_buffer[index]);
	
      	index += 2;;
      }      
      recv_buffer += 400;  
    }
  }
  
  dataInfo.recv_data_len += length * 5 * 4;

  ++threadInfo.current_core;

  if (threadInfo.current_core < threadInfo.cores_in_group)
  {
    // prepare next core data to temp
    data_prepare(&fft_msg);
    
    // send token
    LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

    threadInfo.state = RIGHT_RECVRTOKEN;

    //change core state
	  do_core_state_change();
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND; 
  }
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

// current core is in the same row
void co_recv_col_data()
{
	unsigned short token;
	int i,index,length,j;
	floatv4 data;

	token = threadInfo.token;

  // recv all other row data.
  // get all cores data.
  j = threadInfo.cores_in_group - GET_ROW_CORES(threadInfo.range);
  
  length =  80 * 5;

  index = 0;

  while (0 < j)
  {
    for (i = 0; i < length / 2; ++i)
    { 
      LONG_GETC(data);
      simd_store(data, (float*)&dataInfo.tmp_buffer[index]);

      index += 2;
   
    }
  
    // send tmp data to current core
    send_row_data(dataInfo.tmp_buffer, index, token);

    --j;

    index = 0;
  }

  if (threadInfo.next_col_index != get_token_col_index(token))
      LONG_PUTR(threadInfo.token, threadInfo.next_col_index);

  ++threadInfo.current_core;

  if (threadInfo.current_core < threadInfo.cores_in_group)
  {
    data_prepare(&fft_msg);

    threadInfo.state = RIGHT_RECVRTOKEN;
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND; 
  }
}

// current and comm core or current core in other row 
// first time recv token, co core -> current core.
void cuco_recv_row_token()
{

  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

  threadInfo.state = RIGHT_RECVRDATA;
}

void cuco_recv_row_data()
{
  int i,j,z;
  int length;
  int index;
  int index1;
  int index2;
  FFT_TYPE* recv_buffer;
  floatv4 data;

  // get local row all cores data.       
  index1 = GET_ROW_CORES(threadInfo.range) - 1;

  if (threadInfo.logic_id == threadInfo.token)
  {
    length = 80;

    for (j = 0; j < index1; ++j)
    {
      recv_buffer = dataInfo.recv_buffer;
      index2 = dataInfo.recv_core_seq[j];
      for (z = 0; z < 5; ++z)
      {
        index = index2 * 80; //(local row begin offset)
        for (i = 0; i < length / 2; ++i)
        {
          LONG_GETR(data);
          simd_store(data, (float*)&recv_buffer[index]);

          index += 2;
        }

        recv_buffer += 400;
      }
    }

    dataInfo.recv_data_len += length * 5 * index1;

    LONG_PUTC(threadInfo.token, threadInfo.next_row_index); // col token

    threadInfo.state = RIGHT_RECVCDATA;
  }
  else
  {
    length = index1 * 80 * 5;
    index = dataInfo.tmp_data_index;

    for (i = 0; i < length / 2; ++i)
    {
      LONG_GETR(data);
      simd_store(data, (float*)&dataInfo.tmp_buffer[index]);

      index += 2;
    }

    dataInfo.tmp_data_index = index;
    
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

	if (IS_BEGIN_CORE(threadInfo.range, threadInfo.logic_id) || (400 >= dataInfo.tmp_data_index))
	{
	  send_column_data(dataInfo.tmp_buffer, dataInfo.tmp_data_index, token);
	}
	else
	{
	  send_column_data(dataInfo.tmp_buffer + 400, dataInfo.tmp_data_index - 400, token);

	  send_column_data(dataInfo.tmp_buffer, 400, token);
	}

	// if next row core is not token, send column token.
  if (threadInfo.next_row_index != get_token_row_index(token))
    LONG_PUTC(token, threadInfo.next_col_index);

	++threadInfo.current_core;

	if (threadInfo.current_core < threadInfo.cores_in_group)
  {

	  threadInfo.token = threadInfo.current_core;

    data_prepare(&fft_msg);

    // send token to next core in some row
    LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

    // core state change
    if ((threadInfo.current_core != threadInfo.logic_id) && IN_SAME_ROW(threadInfo.range, threadInfo.current_core))
    {
      do_core_state_change();
      threadInfo.state = RIGHT_RECVRTOKEN;
      return;
    }

    threadInfo.state = RIGHT_RECVRDATA;
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND; 
  }
}

// only current core is comm core
// not put in tmp_buffer,only recv row data may put into tmp_buffer.
void cuco_recv_col_data()
{
  unsigned short token;
	int i,index,index1,length,j,z;
	FFT_TYPE *recv_buffer;
	floatv4 data;
	
	token = threadInfo.token;
	
	// recv all other row data.
	// get all cores data.
	j = GET_ROW_CORES(threadInfo.range) - 1;
		
	length =	80;
	
	for (; j < threadInfo.cores_in_group - 1; ++j)
	{
	  recv_buffer = dataInfo.recv_buffer;
	  index1 = dataInfo.recv_core_seq[j];
	  for (z = 0; z < 5; ++z)
	  {
	    index = index1 * 80;
		  for (i = 0; i < length / 2; ++i)
		  { 
			  LONG_GETC(data);
	      simd_store(data, (float*)&recv_buffer[index]);
			  index += 2;			
		  }

		  recv_buffer += 400;
		}
		
		dataInfo.recv_data_len += 80 * 5;
	}

	++threadInfo.current_core;
	
  if (threadInfo.current_core < threadInfo.cores_in_group)
  {
    data_prepare(&fft_msg);
		
	  // send token
	  LONG_PUTR(threadInfo.current_core, threadInfo.next_col_index);

	  threadInfo.token = threadInfo.current_core;

	  threadInfo.state = RIGHT_RECVRDATA;

	  // change core state change to co
	  if ((threadInfo.current_core != threadInfo.logic_id) && IN_SAME_ROW(threadInfo.range, threadInfo.current_core))
	  {
      do_core_state_change();
      threadInfo.state = RIGHT_RECVRTOKEN;
      return;
    }
  }
  else
  {
    threadInfo.state = RIGHT_ALLEND; 
  }
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

  dataInfo.recv_buffer_size = MAX_PCORE_DATA;
  dataInfo.input_buffer_size = MAX_PCORE_DATA;
  dataInfo.tmp_buffer_size = MAX_PCORE_DATA / 2;

  // cal mode bat or single
  init_core_state();
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

void init_core_state()
{
  switch (threadInfo.core_state)
  {
    case CORE_STATE_N:
    {
      exchangeFunc.recv_rtoken_func = n_recv_row_token;
      exchangeFunc.recv_rdata_func = NULL;
      exchangeFunc.recv_ctoken_func = NULL;
      exchangeFunc.recv_cdata_func = NULL;
    }
    break;
    case CORE_STATE_CU:
    {
      exchangeFunc.recv_rtoken_func = cu_recv_row_token;
      exchangeFunc.recv_rdata_func = cu_recv_row_data;
      exchangeFunc.recv_ctoken_func = NULL;
      exchangeFunc.recv_cdata_func = NULL;
    }
    break;
    case  CORE_STATE_CO:
    {
      exchangeFunc.recv_rtoken_func = co_recv_row_token;
      exchangeFunc.recv_rdata_func = NULL;
      exchangeFunc.recv_ctoken_func = NULL;
      exchangeFunc.recv_cdata_func = co_recv_col_data;
    }
    break;
    case CORE_STATE_CUCO:
    {
      exchangeFunc.recv_rtoken_func = cuco_recv_row_token;
      exchangeFunc.recv_rdata_func = cuco_recv_row_data;
      exchangeFunc.recv_ctoken_func = cuco_recv_col_token;
      exchangeFunc.recv_cdata_func = cuco_recv_col_data;
    }
    break;
    default:
    break;
  }
}

void do_core_state_change()
{
  //if (2 == threadInfo.logic_id)
    //printf("%d %d->", threadInfo.logic_id, threadInfo.core_state);

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

  //if (2 == threadInfo.logic_id)
    //printf("%d %d\n", threadInfo.core_state, threadInfo.current_core);

}

void do_data_exchange()
{
  volatile int state = threadInfo.state;
  while (RIGHT_ALLEND != state)
  {
    switch (state)
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
      return;
    }

    state = threadInfo.state;
  }
}

