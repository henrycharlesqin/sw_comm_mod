#include "slave.h"
#include "simd.h"
#include "type.h"

extern __thread_local THREADINFO threadInfo;

//N == 20 参照 n2fv_20.c  
void n2fv_20_simd(int v,int is,int cly,int cly_sum,FFT_TYPE *input_buf,FFT_TYPE *out_buf)
{
	float KP587785252 = 0.587785252292473129168705954639072768597652438;
	float KP951056516 = 0.951056516295153572116439333379382143405698634;
	float KP250000000 = 0.250000000000000000000000000000000000000000000;
	float KP559016994 = 0.559016994374947424102293417182819058860154590;
	int i,j,k;
	
	FFT_TYPE tmp_buf[40];

	floatv4 Tne = simd_set_floatv4(-1.0,1.0,-1.0,1.0);
	floatv4 Tim;

	
	int offset1;
	int offset2;
	
	for (i=0; i<v; i+=2)	  		
	{	
		floatv4 T3, T1B, Tm, T1i, TG, TN, TO, TH, T13, T16, T1k, T1u, T1v, T1z, T1r;
		floatv4 T1s, T1y, T1a, T1d, T1j, Ti, TD, TB, TL;
		{
			floatv4 T1, T2, T1g, Tk, Tl, T1h;
			
			simd_load(T1,(float *)&input_buf[0 * is + i]);	
			simd_load(T2,(float *)&input_buf[10 * is + i]);
			simd_load(Tk,(float *)&input_buf[5 * is + i]);	
			simd_load(Tl,(float *)&input_buf[15 * is + i]);

			T1g = simd_vadds(T1, T2);
		
			T1h = simd_vadds(Tk, Tl);
			T3 = simd_vsubs(T1, T2);
			T1B = simd_vadds(T1g, T1h);
			Tm = simd_vsubs(Tk, Tl);
			T1i = simd_vsubs(T1g, T1h);
		}
	  {
		  floatv4 T6, T18, Tw, T12, Tz, T15, T9, T1b, Td, T11, Tp, T19, Ts, T1c, Tg;
		  floatv4 T14;
		  {
				floatv4 T4, T5, Tu, Tv;
				
				simd_load(T4,(float *)&input_buf[4 * is + i]);	
				simd_load(T5,(float *)&input_buf[14 * is + i]);
				simd_load(Tu,(float *)&input_buf[13 * is + i]);	
				simd_load(Tv,(float *)&input_buf[3 * is + i]);
				
				T6 = simd_vsubs(T4, T5);
				T18 = simd_vadds(T4, T5);
				
				Tw = simd_vsubs(Tu, Tv);
				T12 = simd_vadds(Tu, Tv);
		  }
		  {
				floatv4 Tx, Ty, T7, T8;
				
				simd_load(Tx,(float *)&input_buf[17 * is + i]);	
				simd_load(Ty,(float *)&input_buf[7 * is + i]);
				simd_load(T7,(float *)&input_buf[16 * is + i]);	
				simd_load(T8,(float *)&input_buf[6 * is + i]);
				
				Tz = simd_vsubs(Tx, Ty);
				T15 = simd_vadds(Tx, Ty);
				T9 = simd_vsubs(T7, T8);
				T1b = simd_vadds(T7, T8);
		  }
		  {
				floatv4 Tb, Tc, Tn, To;
				
				simd_load(Tb,(float *)&input_buf[8 * is + i]);	
				simd_load(Tc,(float *)&input_buf[18 * is + i]);
				simd_load(Tn,(float *)&input_buf[9 * is + i]);	
				simd_load(To,(float *)&input_buf[19 * is + i]);
				
				Td = simd_vsubs(Tb, Tc);
				T11 = simd_vadds(Tb, Tc);
				Tp = simd_vsubs(Tn, To);
				T19 = simd_vadds(Tn, To);
		  }
		  {
				floatv4 Tq, Tr, Te, Tf;
				
				simd_load(Tq,(float *)&input_buf[1 * is + i]);	
				simd_load(Tr,(float *)&input_buf[11 * is + i]);
				simd_load(Te,(float *)&input_buf[12 * is + i]);	
				simd_load(Tf,(float *)&input_buf[2 * is + i]);

				Ts = simd_vsubs(Tq, Tr);
				T1c = simd_vadds(Tq, Tr);
				
				Tg = simd_vsubs(Te, Tf);
				T14 = simd_vadds(Te, Tf);
		  }
		  TG = simd_vsubs(Ts, Tp);
		  TN = simd_vsubs(T6, T9);
		  TO = simd_vsubs(Td, Tg);
		  TH = simd_vsubs(Tz, Tw);
		  T13 = simd_vsubs(T11, T12);
		  T16 = simd_vsubs(T14, T15);
		  T1k = simd_vadds(T13, T16);
		  T1u = simd_vadds(T11, T12);
		  T1v = simd_vadds(T14, T15);
		  T1z = simd_vadds(T1u, T1v);
		  T1r = simd_vadds(T18, T19);
		  T1s = simd_vadds(T1b, T1c);
		  T1y = simd_vadds(T1r, T1s);
		  T1a = simd_vsubs(T18, T19);
		  T1d = simd_vsubs(T1b, T1c);
		  T1j = simd_vadds(T1a, T1d);
		  {
				floatv4 Ta, Th, Tt, TA;
				Ta = simd_vadds(T6, T9);
				Th = simd_vadds(Td, Tg);
				Ti = simd_vadds(Ta, Th);
				TD = (KP559016994)* simd_vsubs(Ta, Th);
				Tt = simd_vadds(Tp, Ts);
				TA = simd_vadds(Tw, Tz);
				TB = simd_vadds(Tt, TA);
				TL = (KP559016994)* simd_vsubs(TA, Tt);
		  }
	  }
		{
			floatv4 T1I, T1J, T1K, T1L, T1N, T1H, Tj, TC;
			Tj = simd_vadds(T3, Ti);
			TC = simd_vadds(Tm, TB);
			Tim = simd_vshff(TC,TC,MM_SHUFFLE(2,3,0,1));
			TC = simd_vmuls(Tim,Tne);
			
			T1H = simd_vsubs(Tj, TC);
			simd_store(T1H,(float *)&tmp_buf[10]); 
			T1I = simd_vadds(Tj, TC);
			simd_store(T1I,(float *)&tmp_buf[30]); 
			{
				floatv4 T1A, T1C, T1D, T1x, T1G, T1t, T1w, T1F, T1E, T1M;
				T1A = KP559016994* simd_vsubs(T1y, T1z);
				T1C = simd_vadds(T1y, T1z);
				T1D = simd_vsubs(T1B,(KP250000000)* T1C);
				T1t = simd_vsubs(T1r, T1s);
				T1w = simd_vsubs(T1u, T1v);
				T1x = simd_vadds((KP951056516)* T1t, (KP587785252)* T1w);
				Tim = simd_vshff(T1x,T1x,MM_SHUFFLE(2,3,0,1));
				T1x = simd_vmuls(Tim,Tne);

				T1G = simd_vsubs((KP951056516)* T1w, (KP587785252)* T1t);
				Tim = simd_vshff(T1G,T1G,MM_SHUFFLE(2,3,0,1));
				T1G = simd_vmuls(Tim,Tne);

				T1J = simd_vadds(T1B, T1C);
				simd_store(T1J,(float *)&tmp_buf[0]); 
				T1F = simd_vsubs(T1D, T1A);
				T1K = simd_vsubs(T1F, T1G);
				simd_store(T1K,(float *)&tmp_buf[16]); 
				T1L = simd_vadds(T1G, T1F);
				simd_store(T1L,(float *)&tmp_buf[24]); 
				T1E = simd_vadds(T1A, T1D);
				T1M = simd_vadds(T1x, T1E);
				simd_store(T1M,(float *)&tmp_buf[8]);
				T1N = simd_vsubs(T1E, T1x);
				simd_store(T1N,(float *)&tmp_buf[32]); 
			}
			{
				floatv4 T1O, T1P, T1R, T1S;
				{
					floatv4 T1n, T1l, T1m, T1f, T1q, T17, T1e, T1p, T1Q, T1o;
					T1n = (KP559016994)* simd_vsubs(T1j, T1k);
					T1l = simd_vadds(T1j, T1k);
					T1m = simd_vsubs(T1i,(KP250000000)* T1l);

					T17 = simd_vsubs(T13, T16);
					T1e = simd_vsubs(T1a, T1d);
					T1f = simd_vsubs(KP951056516* T17,KP587785252* T1e);
					Tim = simd_vshff(T1f,T1f,MM_SHUFFLE(2,3,0,1));
					T1f = simd_vmuls(Tim,Tne);

					T1q = simd_vadds(KP587785252* T17,KP951056516* T1e);
					Tim = simd_vshff(T1q,T1q,MM_SHUFFLE(2,3,0,1));
					T1q = simd_vmuls(Tim,Tne);

					T1O = simd_vadds(T1i, T1l);
					simd_store(T1O,(float *)&tmp_buf[20]); 
					T1p = simd_vadds(T1n, T1m);
					T1P = simd_vsubs(T1p, T1q);
					simd_store(T1P,(float *)&tmp_buf[12]); 
					T1Q = simd_vadds(T1q, T1p);
					simd_store(T1Q,(float *)&tmp_buf[28]);
					T1o = simd_vsubs(T1m, T1n);
					T1R = simd_vadds(T1f, T1o);
					simd_store(T1R,(float *)&tmp_buf[4]); 
					T1S = simd_vsubs(T1o, T1f);
					simd_store(T1S,(float *)&tmp_buf[36]); 
				}
				{
					floatv4 TI, TP, TX, TU, TM, TW, TF, TT, TK, TE;
					TI = simd_vadds(KP587785252* TH,KP951056516* TG);
					TP = simd_vadds(KP587785252* TO,KP951056516* TN);
					TX = simd_vsubs(KP951056516* TO,KP587785252* TN);
					TU = simd_vsubs(KP951056516* TH,KP587785252* TG);

					TK = simd_vsubs(KP250000000* TB, Tm);
					TM = simd_vadds(TK, TL);
					TW = simd_vsubs(TL, TK);
					TE = simd_vsubs(T3,KP250000000* Ti);
					TF = simd_vadds(TD, TE);
					TT = simd_vsubs(TE, TD);
					{
						floatv4 TJ, TQ, T1T, T1U;
						TJ = simd_vadds(TF, TI);
						TQ = simd_vsubs(TM, TP);
						Tim = simd_vshff(TQ,TQ,MM_SHUFFLE(2,3,0,1));
						TQ = simd_vmuls(Tim,Tne);
						T1T = simd_vsubs(TJ, TQ);
						simd_store(T1T,(float *)&tmp_buf[38]);
						T1U = simd_vadds(TJ, TQ);
						simd_store(T1U,(float *)&tmp_buf[2]); 
					}
					{
						floatv4 TZ, T10, T1V, T1W;
						TZ = simd_vadds(TT, TU);
						T10 = simd_vadds(TX, TW);
						Tim = simd_vshff(T10,T10,MM_SHUFFLE(2,3,0,1));
						T10 = simd_vmuls(Tim,Tne);

						T1V = simd_vsubs(TZ, T10);
						simd_store(T1V,(float *)&tmp_buf[26]); 
						T1W = simd_vadds(TZ, T10);
						simd_store(T1W,(float *)&tmp_buf[14]); 
					}
					{
						floatv4 TR, TS, T1X, T1Y;
						TR = simd_vsubs(TF, TI);
						TS = simd_vadds(TP, TM);
						Tim = simd_vshff(TS,TS,MM_SHUFFLE(2,3,0,1));
						TS = simd_vmuls(Tim,Tne);

						T1X = simd_vsubs(TR, TS);
						simd_store(T1X,(float *)&tmp_buf[22]); 
						T1Y = simd_vadds(TR, TS);
						simd_store(T1Y,(float *)&tmp_buf[18]); 
					}
					{
						floatv4 TV, TY, T1Z, T20;
						TV = simd_vsubs(TT, TU);
						TY = simd_vsubs(TW, TX);
						Tim = simd_vshff(TY,TY,MM_SHUFFLE(2,3,0,1));
						TY = simd_vmuls(Tim,Tne);
						T1Z = simd_vsubs(TV, TY);
						simd_store(T1Z,(float *)&tmp_buf[34]); 
						T20 = simd_vadds(TV, TY);
						simd_store(T20,(float *)&tmp_buf[6]); 
					}
				}
			}
		}
		
		if(cly_sum != 1)
		{
			offset1 = 20*v*((cly*v+ i)%cly_sum)+20*((cly*v+i)/cly_sum);
			offset2 = 20*v*((cly*v+(i+1))%cly_sum)+20*((cly*v+(i+1))/cly_sum);
		}
		else
		{
			offset1 = 20*i;
			offset2 = 20*(i+1);
		}
		
		for(j=0, k=0; j<40; j=j+2, k++)
		{
			out_buf[k + offset1] = tmp_buf[j];
			out_buf[k + offset2] = tmp_buf[j+1];	
		}
  }
	return;
}

