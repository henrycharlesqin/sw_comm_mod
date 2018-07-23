#include <stdio.h>
#include "slave.h"
#include "dma.h"
#include "type.h"

extern __thread_local FFT_PARAM slaveParam;
extern __thread_local THREADINFO threadInfo;
extern __thread_local DATAEXCHANGE_INFO dataInfo;
extern __thread_local FFT_PARAM fft_param;
extern __thread_local int thread_id;


// util.c
/*********************************************************/
/*     after init map like this: when cores_per_group = 5 means
a group contains 5 slave cores. this function will allocate logic_id to
current group's slave core.

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

        but the function only show current group's slave core's logic_id.
the others is 0x0FF;
**********************************************************/
static void init_group_map(unsigned short gid, unsigned short cores)
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

static void init_row_range()
{
  	int row_index = 0;
  	int col_index = 0;
  	int i = 0;
  	int j = 0;

  	unsigned char begin,end;
  	unsigned short dis;
  	
	  row_index = CORE_ROW(threadInfo.physical_id);
	  col_index = CORE_COL(threadInfo.physical_id);
	  
		j = 0;
		while (j < MAX_CCORE)
		{
			if (0xFF != threadInfo.core_group_map[row_index][j])
			{
				begin = threadInfo.core_group_map[row_index][j];
				break;
			}
	
			++j;
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

		// next core
		threadInfo.next_col_index = 0x0FF;

		// next row index
		threadInfo.next_row_index = 0x0FF;

    // direction
		if (begin < end)
		{
		  threadInfo.direction = DIR_RIGHT;

		  dis = (unsigned short)(end - begin);

      // next core
		  if (end >= (unsigned char)(threadInfo.logic_id + 1))
		  {
		    threadInfo.next_col_index = col_index + 1;
		  }
		  else
		  {
		  	threadInfo.next_col_index = col_index - dis;
		  }
		}
		else if (begin > end)
		{
		  threadInfo.direction = DIR_LEFT;

		  // range
		  threadInfo.range = SET_16BITS_PARAM(begin, end);

		  dis = (unsigned short)(begin - end);

		  // next core
		  if (begin >= (unsigned char)(threadInfo.logic_id + 1))
		  {
		    threadInfo.next_col_index = col_index - 1;
		  }
		  else
		  {
		  	threadInfo.next_col_index = col_index + dis;
		  }
		}
		else // only one core in group
		{
			threadInfo.direction = ((i == (MAX_CCORE - 1)) ? DIR_LEFT : DIR_RIGHT);
			// next_core_index 0x0FF
		}

		// logic_id
		threadInfo.logic_id = threadInfo.core_group_map[row_index][col_index];

    // comm core
    threadInfo.rows_in_group = 0;
    threadInfo.rows_comm_core = 0x0FF;
    threadInfo.current_row = 0x0FF;
		for (i = 0; i < MAX_RCORE; ++i)
		{
		  if ((0x0FF == threadInfo.core_group_map[i][0]) && (0x0FF == threadInfo.core_group_map[i][MAX_CCORE - 1]))
		    continue;

      if (0x0FF == threadInfo.rows_comm_core)
      {
        if (0x0FF == threadInfo.core_group_map[i][0])
        {
          threadInfo.rows_comm_core = threadInfo.core_group_map[row_index][MAX_CCORE - 1]; // use local row
        }
        else if (0x0FF == threadInfo.core_group_map[i][MAX_CCORE - 1])
        {
          threadInfo.rows_comm_core = threadInfo.core_group_map[row_index][0]; // use local row
        }
      }

      ++threadInfo.rows_in_group;

      if (i == row_index)
      {
        threadInfo.current_row = threadInfo.rows_in_group;
      }
		}

		if (0x0FF == threadInfo.rows_comm_core)
		{
		  threadInfo.rows_comm_core = threadInfo.core_group_map[row_index][MAX_CCORE - 1]; // groups have 8*i cores, used 7 column as default.
		}

		if (threadInfo.logic_id == threadInfo.rows_comm_core)
		{
			row_index = CORE_ROW(threadInfo.physical_id);
	    col_index = CORE_COL(threadInfo.physical_id);
	    if ((row_index + 1) < MAX_RCORE)
	    {
	      if (0x0FF != threadInfo.core_group_map[row_index + 1][col_index])
	      {
	        threadInfo.next_row_index = row_index + 1;
	      }
	      else
	      {
	      	threadInfo.next_row_index = row_index - (threadInfo.rows_in_group - 1);
	      }
	    }
	    else
	    {
	      threadInfo.next_row_index = row_index - (threadInfo.rows_in_group - 1);
	    }
		}

		if (0 == threadInfo.rows_in_group)
		{
		  threadInfo.rows_in_group = 1;
		}
}

unsigned short init_threadinfo(int thread_id, int N)
{
	// cal multi-core read N/n
	int mod = 0;
	int mod1 = 0;
	int quo = 0;
	int cores_per_group = 0;
	int i = 0,j = 0;
	unsigned short start_recvdata_index = 0;
	unsigned short end_recvdata_index = 0;

	unsigned char begin,end;

	threadInfo.logic_id = thread_id;

	mod = N % MAX_PCORE_DATA;
	quo = N / MAX_PCORE_DATA;
	if (0 == mod)
	{
	  cores_per_group = quo;
	  threadInfo.recv_data_rem = 0;
	  dataInfo.input_buffer_size = N / cores_per_group;
	}
	else
	{
		cores_per_group = quo + 1;
		mod1 = N % cores_per_group;
		threadInfo.recv_data_rem = mod1;
		if ( 0 == threadInfo.logic_id)
	  	dataInfo.input_buffer_size = N / cores_per_group + mod1;
	  else
	    dataInfo.input_buffer_size = N / cores_per_group;
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

	// recv_data_range
	j = threadInfo.logic_id;
	if (0  == j)
	{
	  start_recvdata_index = 0;
	  end_recvdata_index = dataInfo.input_buffer_size - 1;
	}
	else
	{
	  start_recvdata_index = j * dataInfo.input_buffer_size + mod1;
	  end_recvdata_index = (j + 1) * dataInfo.input_buffer_size + mod1 - 1;
	}

	threadInfo.recv_data_range = SET_32BITS_PARAM(start_recvdata_index, end_recvdata_index);

	if (1 == threadInfo.rows_in_group)
	{
	  if (0 == threadInfo.logic_id)
	  {
	    threadInfo.core_state = CORE_STATE_CU;
	  }
	  else
	  {
	    threadInfo.core_state = CORE_STATE_N;
	  }
	}
	else // multi rows
	{
	  if (IN_SOME_ROW(threadInfo.range, 0))
	  {
	    if (0 == threadInfo.logic_id)
	    {
	      if (0 == threadInfo.rows_comm_core)
	      {
	        threadInfo.core_state = CORE_STATE_CUCO;
	      }
	      else
	      {
	        threadInfo.core_state = CORE_STATE_CU;
	      }
	    }
	    else
	    {
	      if (threadInfo.logic_id != threadInfo.rows_comm_core)
	      {
	        threadInfo.core_state = CORE_STATE_N;
	      }
	      else
	      {
	        threadInfo.core_state = CORE_STATE_CO;
	      }
	    }
	  }
	  else
	  {
	    if ((threadInfo.logic_id != threadInfo.rows_comm_core) && (0x0FF != threadInfo.rows_comm_core))
	    {
	      threadInfo.core_state = CORE_STATE_N;
	    }
	    else
	    {
	      threadInfo.core_state = CORE_STATE_CUCO;
	    }
	  }
	}
	
	return RET_OK;
	}



