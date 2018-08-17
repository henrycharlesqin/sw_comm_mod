#include "slave.h"
#include "simd.h"
#include "type.h"

extern __thread_local THREADINFO threadInfo;


void t3fv_20_simd(FFT_TYPE *ri, const FFT_TYPE *W, int rs, int mb, int me, int ms)
{
	float KP587785252 = 0.587785252292473129168705954639072768597652438;
	float KP951056516 = 0.951056516295153572116439333379382143405698634;
	float KP250000000 = 0.250000000000000000000000000000000000000000000;
	float KP559016994 = 0.559016994374947424102293417182819058860154590;

	int i,j,k;
	int z;

	floatv4 Tne = simd_set_floatv4(-1.0,1.0,-1.0,1.0);
	floatv4 Tim;
	floatv4 Ttemp1, Ttemp;
	floatv4 Trr, Tii;

	for (i=mb; i<me; i+=2)
	{	
		floatv4 T2, T8, T9, TA, T3, Tc, T4, TZ, T18, Tl, Tq, Tx, TU, Td, Te;
		floatv4 T15, Ti, Tt, TJ;
		
		//T2 = simd_set_floatv4(W[0+4*i].re,W[0+4*i].im,W[0+4*(i+1)].re,W[0+4*(i+1)].im);
		//T8 = simd_set_floatv4(W[1+4*i].re,W[1+4*i].im,W[1+4*(i+1)].re,W[1+4*(i+1)].im);
		//T3 = simd_set_floatv4(W[2+4*i].re,W[2+4*i].im,W[2+4*(i+1)].re,W[2+4*(i+1)].im);
		//Td = simd_set_floatv4(W[3+4*i].re,W[3+4*i].im,W[3+4*(i+1)].re,W[3+4*(i+1)].im);
		z = (0 + 4*i);

		simd_load(T2, (float *)&W[z]);
		simd_load(T8, (float *)&W[z+2]);
		simd_load(T3, (float *)&W[z+4]);
		simd_load(Td, (float *)&W[z+6]);
				
		Trr = simd_vshff(T2,T2,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T2,T2,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T9 = simd_vmas(Tii,Ttemp1,Ttemp);
		TA = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		Ttemp = simd_vmuls(Trr,T3);
		Tim = simd_vshff(T3,T3,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T4 = simd_vmas(Tii,Ttemp1,Ttemp);
		Tq = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		Trr = simd_vshff(T9,T9,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T9,T9,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T3);
		Tim = simd_vshff(T3,T3,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		TZ = simd_vmas(Tii,Ttemp1,Ttemp);
		Tc = simd_vnmas(Tii,Ttemp1,Ttemp);
		

		Trr = simd_vshff(TA,TA,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(TA,TA,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T3);
		Tim = simd_vshff(T3,T3,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		TU = simd_vmas(Tii,Ttemp1,Ttemp);
		T18 = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		Trr = simd_vshff(T8,T8,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T8,T8,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T3);
		Tim = simd_vshff(T3,T3,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Tx = simd_vmas(Tii,Ttemp1,Ttemp);
		Tl = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		
		Trr = simd_vshff(Tc,Tc,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(Tc,Tc,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,Td);
		Tim = simd_vshff(Td,Td,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Te = simd_vnmas(Tii,Ttemp1,Ttemp);

		Trr = simd_vshff(TA,TA,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(TA,TA,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,Td);
		Tim = simd_vshff(Td,Td,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T15 = simd_vnmas(Tii,Ttemp1,Ttemp);

		Trr = simd_vshff(T8,T8,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T8,T8,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,Td);
		Tim = simd_vshff(Td,Td,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Ti = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		Trr = simd_vshff(T2,T2,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T2,T2,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,Td);
		Tim = simd_vshff(Td,Td,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Tt = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		Trr = simd_vshff(T9,T9,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T9,T9,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,Td);
		Tim = simd_vshff(Td,Td,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		TJ = simd_vnmas(Tii,Ttemp1,Ttemp);

		{
		  floatv4 T7, TM, T1U, T2d, T1i, T1p, T1q, T1j, Tp, TE, TF, T26, T27, T2b, T1M;
		  floatv4 T1P, T1V, TY, T1c, T1d, T23, T24, T2a, T1F, T1I, T1W, TG, T1e;
		   {
				floatv4 T1, TL, T6, TI, TK, T5, TH, T1S, T1T;
			
				simd_load(T1,(float *)&ri[rs*0 + i*ms]);
				simd_load(TK,(float *)&ri[rs*15 + i*ms]);
				simd_load(T5,(float *)&ri[rs*10+ i*ms]);
				simd_load(TH,(float *)&ri[rs*5 + i*ms]);
				
				Trr = simd_vshff(TJ,TJ,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(TJ,TJ,MM_SHUFFLE(3,3,1,1));

				Ttemp = simd_vmuls(Trr,TK);
				Tim = simd_vshff(TK,TK,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				TL = simd_vnmas(Tii,Ttemp1,Ttemp);

				Trr = simd_vshff(T4,T4,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(T4,T4,MM_SHUFFLE(3,3,1,1));

				Ttemp = simd_vmuls(Trr,T5);
				Tim = simd_vshff(T5,T5,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				T6 = simd_vnmas(Tii,Ttemp1,Ttemp);

				Trr = simd_vshff(Tc,Tc,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(Tc,Tc,MM_SHUFFLE(3,3,1,1));

				Ttemp = simd_vmuls(Trr,TH);
				Tim = simd_vshff(TH,TH,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				TI = simd_vnmas(Tii,Ttemp1,Ttemp);

				T7 = simd_vsubs(T1, T6);
				TM = simd_vsubs(TI, TL);
				T1S = simd_vadds(T1, T6);
				T1T = simd_vadds(TI, TL);
				T1U = simd_vsubs(T1S, T1T);
				T2d = simd_vadds(T1S, T1T);
		    }
		    {
				floatv4 Th, T1K, T14, T1E, T1b, T1H, To, T1N, Tw, T1D, TR, T1L, TX, T1O, TD;
				floatv4 T1G;
				{
					floatv4 Tb, Tg, Ta, Tf;

					simd_load(Ta,(float *)&ri[rs*4 + i*ms]);
					simd_load(Tf,(float *)&ri[rs*14 + i*ms]);
					
					Trr = simd_vshff(T9,T9,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T9,T9,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Ta);
					Tim = simd_vshff(Ta,Ta,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tb = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(Te,Te,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Te,Te,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Tf);
					Tim = simd_vshff(Tf,Tf,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tg = simd_vnmas(Tii,Ttemp1,Ttemp);


					Th = simd_vsubs(Tb, Tg);
					T1K = simd_vadds(Tb, Tg);
				}
				{
					floatv4 T11, T13, T10, T12;
					
					simd_load(T10,(float *)&ri[rs*13 + i*ms]);
					simd_load(T12,(float *)&ri[rs*3 + i*ms]);

					Trr = simd_vshff(TZ,TZ,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TZ,TZ,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,T10);
					Tim = simd_vshff(T10,T10,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T11 = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(T8,T8,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T8,T8,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,T12);
					Tim = simd_vshff(T12,T12,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T13 = simd_vnmas(Tii,Ttemp1,Ttemp);

					T14 = simd_vsubs(T11, T13);
					T1E = simd_vadds(T11, T13);
				}
				{
					floatv4 T17, T1a, T16, T19;

					simd_load(T16,(float *)&ri[rs*17 + i*ms]);
					simd_load(T19,(float *)&ri[rs*7 + i*ms]);
					
					Trr = simd_vshff(T15,T15,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T15,T15,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,T16);
					Tim = simd_vshff(T16,T16,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T17 = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(T18,T18,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T18,T18,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,T19);
					Tim = simd_vshff(T19,T19,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1a = simd_vnmas(Tii,Ttemp1,Ttemp);

					T1b = simd_vsubs(T17, T1a);
					T1H = simd_vadds(T17, T1a);
				}
				{
					floatv4 Tk, Tn, Tj, Tm;
					
					simd_load(Tj,(float *)&ri[rs*16 + i*ms]);
					simd_load(Tm,(float *)&ri[rs*6 + i*ms]);
					
					Trr = simd_vshff(Ti,Ti,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Ti,Ti,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Tj);
					Tim = simd_vshff(Tj,Tj,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tk = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(Tl,Tl,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tl,Tl,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Tm);
					Tim = simd_vshff(Tm,Tm,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tn = simd_vnmas(Tii,Ttemp1,Ttemp);

					To = simd_vsubs(Tk, Tn);
					T1N = simd_vadds(Tk, Tn);
				}
				{
					floatv4 Ts, Tv, Tr, Tu;
					
					simd_load(Tr,(float *)&ri[rs*8 + i*ms]);
					simd_load(Tu,(float *)&ri[rs*18+ i*ms]);
					
					Trr = simd_vshff(Tq,Tq,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tq,Tq,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Tr);
					Tim = simd_vshff(Tr,Tr,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Ts = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(Tt,Tt,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tt,Tt,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Tu);
					Tim = simd_vshff(Tu,Tu,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tv = simd_vnmas(Tii,Ttemp1,Ttemp);

					Tw = simd_vsubs(Ts, Tv);
					T1D = simd_vadds(Ts, Tv);
				}
				{
					floatv4 TO, TQ, TN, TP;

					simd_load(TN,(float *)&ri[rs*9+ i*ms]);
					simd_load(TP,(float *)&ri[rs*19+ i*ms]);
					
					Trr = simd_vshff(T3,T3,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T3,T3,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,TN);
					Tim = simd_vshff(TN,TN,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TO = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(Td,Td,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Td,Td,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,TP);
					Tim = simd_vshff(TP,TP,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TQ = simd_vnmas(Tii,Ttemp1,Ttemp);

					TR = simd_vsubs(TO, TQ);
					T1L = simd_vadds(TO, TQ);
				}
				{
					floatv4 TT, TW, TS, TV;
				
				  simd_load(TS,(float *)&ri[rs*1+ i*ms]);
					simd_load(TV,(float *)&ri[rs*11+ i*ms]);

					Trr = simd_vshff(T2,T2,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T2,T2,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,TS);
					Tim = simd_vshff(TS,TS,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TT = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(TU,TU,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TU,TU,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,TV);
					Tim = simd_vshff(TV,TV,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TW = simd_vnmas(Tii,Ttemp1,Ttemp);

					TX = simd_vsubs(TT, TW);
					T1O = simd_vadds(TT, TW);
				}
				{
					floatv4 Tz, TC, Ty, TB;

					simd_load(Ty,(float *)&ri[rs*12+ i*ms]);
					simd_load(TB,(float *)&ri[rs*2+ i*ms]);
					
					Trr = simd_vshff(Tx,Tx,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tx,Tx,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,Ty);
					Tim = simd_vshff(Ty,Ty,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tz = simd_vnmas(Tii,Ttemp1,Ttemp);

					Trr = simd_vshff(TA,TA,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TA,TA,MM_SHUFFLE(3,3,1,1));

					Ttemp = simd_vmuls(Trr,TB);
					Tim = simd_vshff(TB,TB,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TC = simd_vnmas(Tii,Ttemp1,Ttemp);

					TD = simd_vsubs(Tz, TC);
					T1G = simd_vadds(Tz, TC);
				}
				
				T1i = simd_vsubs(TX, TR);
				T1p = simd_vsubs(Th, To);
				T1q = simd_vsubs(Tw, TD);
				T1j = simd_vsubs(T1b, T14);
				Tp = simd_vadds(Th, To);
				TE = simd_vadds(Tw, TD);
				TF = simd_vadds(Tp, TE);
				T26 = simd_vadds(T1D, T1E);
				T27 = simd_vadds(T1G, T1H);
				T2b = simd_vadds(T26, T27);
				T1M = simd_vsubs(T1K, T1L);
				T1P = simd_vsubs(T1N, T1O);
				T1V = simd_vadds(T1M, T1P);
				TY = simd_vadds(TR, TX);
				T1c = simd_vadds(T14, T1b);
				T1d = simd_vadds(TY, T1c);
				T23 = simd_vadds(T1K, T1L);
				T24 = simd_vadds(T1N, T1O);
				T2a = simd_vadds(T23, T24);
				T1F = simd_vsubs(T1D, T1E);
				T1I = simd_vsubs(T1G, T1H);
				T1W = simd_vadds(T1F, T1I);
		  }
			
		  TG = simd_vadds(T7, TF);
		  T1e = simd_vadds(TM, T1d);
			Tim = simd_vshff(T1e,T1e,MM_SHUFFLE(2,3,0,1));
			T1e = simd_vmuls(Tim,Tne);
			
			simd_store(simd_vsubs(TG, T1e),(float *)&ri[rs*5+ i*ms]);
			simd_store(simd_vadds(TG, T1e),(float *)&ri[rs*15 + i*ms]);
		  {
				floatv4 T2c, T2e, T2f, T29, T2i, T25, T28, T2h, T2g;
				T2c = (KP559016994)* simd_vsubs(T2a, T2b);
				T2e = simd_vadds(T2a, T2b);
				T2f = simd_vsubs(T2d,(KP250000000)* T2e);
				T25 = simd_vsubs(T23, T24);
				T28 = simd_vsubs(T26, T27);
				T29 = simd_vadds((KP587785252)* T28,(KP951056516)* T25);
				Tim = simd_vshff(T29,T29,MM_SHUFFLE(2,3,0,1));
				T29 = simd_vmuls(Tim,Tne);

				T2i = simd_vsubs((KP951056516)* T28,(KP587785252)* T25);
				Tim = simd_vshff(T2i,T2i,MM_SHUFFLE(2,3,0,1));
				T2i = simd_vmuls(Tim,Tne);

				T2h = simd_vsubs(T2f, T2c);
				T2g = simd_vadds(T2c, T2f);

				simd_store(simd_vadds(T2d, T2e),(float *)&ri[rs*0+ i*ms]);
				simd_store(simd_vsubs(T2h, T2i),(float *)&ri[rs*8 + i*ms]);	
				simd_store(simd_vadds(T2i, T2h),(float *)&ri[rs*12+ i*ms]);
				simd_store(simd_vadds(T29, T2g),(float *)&ri[rs*4 + i*ms]);	
				simd_store(simd_vsubs(T2g, T29),(float *)&ri[rs*16+ i*ms]);	
			}
		  {
				floatv4 T1Z, T1X, T1Y, T1R, T22, T1J, T1Q, T21, T20;
				T1Z = (KP559016994)* simd_vsubs(T1V, T1W);
				T1X = simd_vadds(T1V, T1W);
				T1Y = simd_vsubs(T1U,(KP250000000)* T1X);
				T1J = simd_vsubs(T1F, T1I);
				T1Q = simd_vsubs(T1M, T1P);
				T1R = simd_vsubs((KP951056516)* T1J,(KP587785252)* T1Q);
				Tim = simd_vshff(T1R,T1R,MM_SHUFFLE(2,3,0,1));
				T1R = simd_vmuls(Tim,Tne);
				T22 = simd_vadds((KP587785252)* T1J,(KP951056516)* T1Q);
				Tim = simd_vshff(T22,T22,MM_SHUFFLE(2,3,0,1));
				T22 = simd_vmuls(Tim,Tne);

				T21 = simd_vadds(T1Z, T1Y);
				T20 = simd_vsubs(T1Y, T1Z);
							
				simd_store(simd_vadds(T1U, T1X),(float *)&ri[rs*10+ i*ms]);
				simd_store(simd_vsubs(T21, T22),(float *)&ri[rs*6 + i*ms]);	
				simd_store(simd_vadds(T22, T21),(float *)&ri[rs*14+ i*ms]);
				simd_store(simd_vadds(T1R, T20),(float *)&ri[rs*2+ i*ms]);	
				simd_store(simd_vsubs(T20, T1R),(float *)&ri[rs*18+ i*ms]); 
		  }
		  {
				floatv4 T1k, T1r, T1z, T1w, T1o, T1y, T1h, T1v;
				T1k = simd_vadds((KP587785252)* T1j,(KP951056516)* T1i);
				T1r = simd_vadds((KP587785252)* T1q,(KP951056516)* T1p);
				T1z = simd_vsubs((KP951056516)* T1q,(KP587785252)* T1p);
				T1w = simd_vsubs((KP951056516)* T1j,(KP587785252)* T1i);
				{
					floatv4 T1m, T1n, T1f, T1g;
					T1m = simd_vsubs((KP250000000)* T1d, TM);
					T1n = (KP559016994)* simd_vsubs(T1c, TY);
					T1o = simd_vadds(T1m, T1n);
					T1y = simd_vsubs(T1n, T1m);
					T1f = (KP559016994)* simd_vsubs(Tp, TE);
					T1g = simd_vsubs(T7,(KP250000000)* TF);
					T1h = simd_vadds(T1f, T1g);
					T1v = simd_vsubs(T1g, T1f);
				}
				{
					floatv4 T1l, T1s, T1B, T1C;
					T1l = simd_vadds(T1h, T1k);
					T1s = simd_vsubs(T1o, T1r);
					Tim = simd_vshff(T1s,T1s,MM_SHUFFLE(2,3,0,1));
					T1s = simd_vmuls(Tim,Tne);

					T1B = simd_vadds(T1v, T1w);
					T1C = simd_vadds(T1z, T1y);
					Tim = simd_vshff(T1C,T1C,MM_SHUFFLE(2,3,0,1));
					T1C = simd_vmuls(Tim,Tne);
					
					simd_store(simd_vsubs(T1l, T1s),(float *)&ri[rs*19+ i*ms]);
					simd_store(simd_vadds(T1l, T1s),(float *)&ri[rs*1 + i*ms]);	
					simd_store(simd_vsubs(T1B, T1C),(float *)&ri[rs*13+ i*ms]);
					simd_store(simd_vadds(T1B, T1C),(float *)&ri[rs*7+ i*ms]);
				}
				{
					floatv4 T1t, T1u, T1x, T1A;
					T1t = simd_vsubs(T1h, T1k);
					T1u = simd_vadds(T1r, T1o);
					Tim = simd_vshff(T1u,T1u,MM_SHUFFLE(2,3,0,1));
					T1u = simd_vmuls(Tim,Tne);

					T1x = simd_vsubs(T1v, T1w);
					T1A = simd_vsubs(T1y, T1z);
					Tim = simd_vshff(T1A,T1A,MM_SHUFFLE(2,3,0,1));
					T1A = simd_vmuls(Tim,Tne);

					simd_store(simd_vsubs(T1t, T1u),(float *)&ri[rs*11+ i*ms]);
					simd_store(simd_vadds(T1t, T1u),(float *)&ri[rs*9+ i*ms]);	
					simd_store(simd_vsubs(T1x, T1A),(float *)&ri[rs*17+ i*ms]);
					simd_store(simd_vadds(T1x, T1A),(float *)&ri[rs*3+ i*ms]);	 
				}
		  }
	  }
	}
	return ;
}
#if 0
static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 3),
     VTW(0, 9),
     VTW(0, 19),
     {TW_NEXT, VL, 0}
};
#endif


