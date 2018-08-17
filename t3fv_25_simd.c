#include "slave.h"
#include "simd.h"
#include "type.h"

extern __thread_local THREADINFO threadInfo;

extern __thread_local int thread_id;
void t3fv_25_simd(FFT_TYPE *ri, const FFT_TYPE *W, int rs, int mb, int me, int ms)
{
  float KP998026728 = 0.998026728428271561952336806863450553336905220 ;
  float KP125581039 = 0.125581039058626752152356449131262266244969664 ;
  float KP1_996053456 = 1.996053456856543123904673613726901106673810439 ;
  float KP062790519 = 0.062790519529313376076178224565631133122484832 ;
    float KP809016994 = 0.809016994374947424102293417182819058860154590 ;
    float KP309016994 = 0.309016994374947424102293417182819058860154590 ;
    float KP1_369094211 = 1.369094211857377347464566715242418539779038465 ;
    float KP728968627 = 0.728968627421411523146730319055259111372571664 ;
    float KP963507348 = 0.963507348203430549974383005744259307057084020 ;
    float KP876306680 = 0.876306680043863587308115903922062583399064238 ;
    float KP497379774 = 0.497379774329709576484567492012895936835134813 ;
    float KP968583161 = 0.968583161128631119490168375464735813836012403 ;
    float KP684547105 = 0.684547105928688673732283357621209269889519233 ;
    float KP1_457937254 = 1.457937254842823046293460638110518222745143328 ;
    float KP481753674 = 0.481753674101715274987191502872129653528542010 ;
    float KP1_752613360 = 1.752613360087727174616231807844125166798128477 ;
    float KP248689887 = 0.248689887164854788242283746006447968417567406 ;
    float KP1_937166322 = 1.937166322257262238980336750929471627672024806 ;
    float KP992114701 = 0.992114701314477831049793042785778521453036709 ;
    float KP250666467 = 0.250666467128608490746237519633017587885836494 ;
    float KP425779291 = 0.425779291565072648862502445744251703979973042 ;
    float KP1_809654104 = 1.809654104932039055427337295865395187940827822 ;
    float KP1_274847979 = 1.274847979497379420353425623352032390869834596 ;
    float KP770513242 = 0.770513242775789230803009636396177847271667672 ;
    float KP844327925 = 0.844327925502015078548558063966681505381659241 ;
    float KP1_071653589 = 1.071653589957993236542617535735279956127150691 ;
    float KP125333233 = 0.125333233564304245373118759816508793942918247 ;
    float KP1_984229402 = 1.984229402628955662099586085571557042906073418 ;
    float KP904827052 = 0.904827052466019527713668647932697593970413911 ;
    float KP851558583 = 0.851558583130145297725004891488503407959946084 ;
    float KP637423989 = 0.637423989748689710176712811676016195434917298 ;
    float KP1_541026485 = 1.541026485551578461606019272792355694543335344;
    float KP535826794 = 0.535826794978996618271308767867639978063575346;
    float KP1_688655851 = 1.688655851004030157097116127933363010763318483;
    float KP293892626 = 0.293892626146236564584352977319536384298826219 ;
    float KP475528258 = 0.475528258147576786058219666689691071702849317 ;
    float KP587785252 = 0.587785252292473129168705954639072768597652438 ;
    float KP951056516 = 0.951056516295153572116439333379382143405698634 ;
    float KP250000000 = 0.250000000000000000000000000000000000000000000 ;
    float KP559016994 = 0.559016994374947424102293417182819058860154590 ;
    
	int i,j,k,z;
	FFT_TYPE tmp_buf[50];

	floatv4 Tne = simd_set_floatv4(-1.0,1.0,-1.0,1.0);
	floatv4 Tim;
	floatv4 Ttemp1, Ttemp;
	floatv4 Trr, Tii;

	
	for (i=mb;i<me;i+=2)
	{	
		floatv4 T1, T4, T2, T3, TA, Td, Tp, Tw, Tx, T1G, T1j, T5, T1c, T8, T9;
	    floatv4 Ts, T1J, Tg, T1C, T1m, TX, TB, T1f, TU;
			
		//T1 = simd_set_floatv4(W[0+4*i].re,W[0+4*i].im,W[0+4*(i+1)].re,W[0+4*(i+1)].im);
		//T2 = simd_set_floatv4(W[1+4*i].re,W[1+4*i].im,W[1+4*(i+1)].re,W[1+4*(i+1)].im);
		//T4 = simd_set_floatv4(W[2+4*i].re,W[2+4*i].im,W[2+4*(i+1)].re,W[2+4*(i+1)].im);
		//T8 = simd_set_floatv4(W[3+4*i].re,W[3+4*i].im,W[3+4*(i+1)].re,W[3+4*(i+1)].im);
		z = 0 + 4*i;

		simd_load(T1, (float *)&W[z]);
		simd_load(T2, (float *)&W[z+2]);
		simd_load(T4, (float *)&W[z+4]);
		simd_load(T8, (float *)&W[z+6]);

		//if (0 == threadInfo.logic_id)
		//{
		//  printf("%d ", i);
		//  simd_print_floatv4(T1);
		//  simd_print_floatv4(T2);
		//  simd_print_floatv4(T4);
		//  simd_print_floatv4(T8);
		//}
		
		Trr = simd_vshff(T1,T1,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T1,T1,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T2);
		Tim = simd_vshff(T2,T2,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T3 = simd_vmas(Tii,Ttemp1,Ttemp);
		Tw = simd_vnmas(Tii,Ttemp1,Ttemp);
				    
		Trr = simd_vshff(T1,T1,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T1,T1,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T4);
		Tim = simd_vshff(T4,T4,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Td = simd_vmas(Tii,Ttemp1,Ttemp);
		TA = simd_vnmas(Tii,Ttemp1,Ttemp);
		
		Trr = simd_vshff(T2,T2,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T2,T2,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T4);
		Tim = simd_vshff(T4,T4,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T1j = simd_vmas(Tii,Ttemp1,Ttemp);
		Tp = simd_vnmas(Tii,Ttemp1,Ttemp);

		Trr = simd_vshff(Tw,Tw,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(Tw,Tw,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T4);
		Tim = simd_vshff(T4,T4,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Tx = simd_vmas(Tii,Ttemp1,Ttemp);
		T1c = simd_vnmas(Tii,Ttemp1,Ttemp);   
		
	    Trr = simd_vshff(T3,T3,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T3,T3,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T4);
		Tim = simd_vshff(T4,T4,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T1G = simd_vmas(Tii,Ttemp1,Ttemp);
		T5 = simd_vnmas(Tii,Ttemp1,Ttemp);   
	    		   
		Trr = simd_vshff(T3,T3,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T3,T3,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T9 = simd_vnmas(Tii,Ttemp1,Ttemp);    
		
		Trr = simd_vshff(T2,T2,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T2,T2,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Ts = simd_vnmas(Tii,Ttemp1,Ttemp);   
		
	    Trr = simd_vshff(Tp,Tp,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(Tp,Tp,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T1J = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(T4,T4,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T4,T4,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		Tg = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(T1,T1,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T1,T1,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T1C = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(T1c,T1c,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T1c,T1c,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T1m = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(T5,T5,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(T5,T5,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		TX = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(TA,TA,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(TA,TA,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		TB = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(Tw,Tw,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(Tw,Tw,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		T1f = simd_vnmas(Tii,Ttemp1,Ttemp);
		
	    Trr = simd_vshff(Td,Td,MM_SHUFFLE(2,2,0,0));
		Tii = simd_vshff(Td,Td,MM_SHUFFLE(3,3,1,1));
		Ttemp = simd_vmuls(Trr,T8);
		Tim = simd_vshff(T8,T8,MM_SHUFFLE(2,3,0,1));
		Ttemp1 = simd_vmuls(Tim,Tne);
		TU = simd_vnmas(Tii,Ttemp1,Ttemp);   
	
		{
		    floatv4 Tl, Tk, Tm, Tn, T20, T2R, T22, T1V, T2K, T1S, T3A, T2L, TN, T2G, TK;
		    floatv4 T3w, T2H, T19, T2D, T16, T3x, T2E, T1y, T2N, T1v, T3z, T2O;
		    {
				floatv4 Tf, Ti, Tj, T7, Tb, Tc, T21;
				
				simd_load(Tl,(float *)&ri[rs*0 + i*ms]);	
					
				{
					floatv4 Te, Th, T6, Ta;
					
					simd_load(Te,(float *)&ri[rs*10 + i*ms]);
					simd_load(Th,(float *)&ri[rs*15 + i*ms]);
					simd_load(T6,(float *)&ri[rs*5+ i*ms]);
					simd_load(Ta,(float *)&ri[rs*20 + i*ms]);		

					//if (0 == threadInfo.logic_id)
					//{
		      //  simd_print_floatv4(Te);
		      //  simd_print_floatv4(Th);
		      //  simd_print_floatv4(T6);
		      //  simd_print_floatv4(Ta);
		      //}
				
					Trr = simd_vshff(Td,Td,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Td,Td,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,Te);
					Tim = simd_vshff(Te,Te,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tf = simd_vnmas(Tii,Ttemp1,Ttemp);
		
					Trr = simd_vshff(Tg,Tg,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tg,Tg,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,Th);
					Tim = simd_vshff(Th,Th,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Ti = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Tj = simd_vadds(Tf, Ti);
					
					Trr = simd_vshff(T5,T5,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T5,T5,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T6);
					Tim = simd_vshff(T6,T6,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T7 = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(T9,T9,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T9,T9,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,Ta);
					Tim = simd_vshff(Ta,Ta,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tb = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Tc = simd_vadds(T7, Tb);
					
				
				}
				
				Tk = (KP559016994)* simd_vsubs(Tc, Tj);
				Tm = simd_vadds(Tc, Tj);

				Tn = simd_vsubs(Tl,(KP250000000)* Tm);
				T20 = simd_vsubs(T7, Tb);
				T21 = simd_vsubs(Tf, Ti);
				T2R = (KP951056516)* T21;
				T22 = simd_vadds((KP587785252)* T21,(KP951056516)* T20);
		    }
		    {
				floatv4 T1P, T1I, T1L, T1M, T1B, T1E, T1F, T1O;
				
				simd_load(T1O,(float *)&ri[rs*3 + i*ms]);		
				
				Trr = simd_vshff(T2,T2,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(T2,T2,MM_SHUFFLE(3,3,1,1));
				Ttemp = simd_vmuls(Trr,T1O);
				Tim = simd_vshff(T1O,T1O,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				T1P = simd_vnmas(Tii,Ttemp1,Ttemp);
				
				{
					floatv4 T1H, T1K, T1A, T1D;
					
					simd_load(T1H,(float *)&ri[rs*13 + i*ms]);
					simd_load(T1K,(float *)&ri[rs*18 + i*ms]);
					simd_load(T1A,(float *)&ri[rs*8+ i*ms]);
					simd_load(T1D,(float *)&ri[rs*23 + i*ms]);		
					
					Trr = simd_vshff(T1G,T1G,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1G,T1G,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1H);
					Tim = simd_vshff(T1H,T1H,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1I = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(T1J,T1J,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1J,T1J,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1K);
					Tim = simd_vshff(T1K,T1K,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1L = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					T1M = simd_vadds(T1I, T1L);
					
					Trr = simd_vshff(TA,TA,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TA,TA,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1A);
					Tim = simd_vshff(T1A,T1A,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1B = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(T1C,T1C,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1C,T1C,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1D);
					Tim = simd_vshff(T1D,T1D,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1E = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					T1F = simd_vadds(T1B, T1E);
				}
				{
					floatv4 T1T, T1U, T1N, T1Q, T1R;
					T1T = simd_vsubs(T1B, T1E);
					T1U = simd_vsubs(T1I, T1L);
					T1V = simd_vadds((KP293892626)* T1U,(KP475528258)* T1T);
					T2K = simd_vsubs((KP475528258)* T1U,(KP293892626)* T1T);
					T1N = (KP559016994)* simd_vsubs(T1F, T1M);
					T1Q = simd_vadds(T1F, T1M);
					T1R = simd_vsubs(T1P,(KP250000000)* T1Q);
					T1S = simd_vadds(T1N, T1R);
					T3A = simd_vadds(T1P, T1Q);
					T2L = simd_vsubs(T1R, T1N);
					
				}
		    }
		    {
				floatv4 TH, Tz, TD, TE, Tr, Tu, Tv, TG;
				
				simd_load(TG,(float *)&ri[rs*1 + i*ms]);	
				
				Trr = simd_vshff(T1,T1,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(T1,T1,MM_SHUFFLE(3,3,1,1));
				Ttemp = simd_vmuls(Trr,TG);
				Tim = simd_vshff(TG,TG,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				TH = simd_vnmas(Tii,Ttemp1,Ttemp);
				
				
				{
					floatv4 Ty, TC, Tq, Tt;
					
					simd_load(Ty,(float *)&ri[rs*11 + i*ms]);
					simd_load(TC,(float *)&ri[rs*16 + i*ms]);
					simd_load(Tq,(float *)&ri[rs*6 + i*ms]);
					simd_load(Tt,(float *)&ri[rs*21 + i*ms]);
					
					
					Trr = simd_vshff(Tx,Tx,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tx,Tx,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,Ty);
					Tim = simd_vshff(Ty,Ty,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tz = simd_vnmas(Tii,Ttemp1,Ttemp);
				
					Trr = simd_vshff(TB,TB,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TB,TB,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,TC);
					Tim = simd_vshff(TC,TC,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TD = simd_vnmas(Tii,Ttemp1,Ttemp);
				
					TE = simd_vadds(Tz, TD);
					
					Trr = simd_vshff(Tp,Tp,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Tp,Tp,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,Tq);
					Tim = simd_vshff(Tq,Tq,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tr = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(Ts,Ts,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(Ts,Ts,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,Tt);
					Tim = simd_vshff(Tt,Tt,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					Tu = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Tv = simd_vadds(Tr, Tu);
				}
				{
					floatv4 TL, TM, TF, TI, TJ;
					TL = simd_vsubs(Tr, Tu);
					TM = simd_vsubs(Tz, TD);
					TN = simd_vadds((KP293892626)* TM,(KP475528258)* TL);
					T2G = simd_vsubs((KP475528258)* TM,(KP293892626)* TL);
					TF = (KP559016994)* simd_vsubs(Tv, TE);
					TI = simd_vadds(Tv, TE);
					TJ = simd_vsubs(TH,(KP250000000)* TI);
					TK = simd_vadds(TF, TJ);
					T3w = simd_vadds(TH, TI);
					T2H = simd_vsubs(TJ, TF);
				}
		    }
		    {
				floatv4 T13, TW, TZ, T10, TQ, TS, TT, T12;
				
				simd_load(T12,(float *)&ri[rs*4 + i*ms]);
				
				Trr = simd_vshff(T3,T3,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(T3,T3,MM_SHUFFLE(3,3,1,1));
				Ttemp = simd_vmuls(Trr,T12);
				Tim = simd_vshff(T12,T12,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				T13 = simd_vnmas(Tii,Ttemp1,Ttemp);
			 
			
				{
					floatv4 TV, TY, TP, TR;
					
					simd_load(TV,(float *)&ri[rs*14 + i*ms]);
					simd_load(TY,(float *)&ri[rs*19 + i*ms]);
					simd_load(TP,(float *)&ri[rs*9+ i*ms]);
					simd_load(TR,(float *)&ri[rs*24 + i*ms]);	
					
					Trr = simd_vshff(TU,TU,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TU,TU,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,TV);
					Tim = simd_vshff(TV,TV,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TW = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(TX,TX,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(TX,TX,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,TY);
					Tim = simd_vshff(TY,TY,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TZ = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					T10 = simd_vadds(TW, TZ);
					
					Trr = simd_vshff(T4,T4,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T4,T4,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,TP);
					Tim = simd_vshff(TP,TP,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TQ = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(T8,T8,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T8,T8,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,TR);
					Tim = simd_vshff(TR,TR,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					TS = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					TT = simd_vadds(TQ, TS);
				}
				{
					floatv4 T17, T18, T11, T14, T15;
					T17 = simd_vsubs(TQ, TS);
					T18 = simd_vsubs(TW, TZ);
					T19 = simd_vadds((KP293892626)* T18,(KP475528258)* T17);
					T2D = simd_vsubs((KP475528258)* T18,(KP293892626)* T17);
					T11 = (KP559016994)* simd_vsubs(TT, T10);
					T14 = simd_vadds(TT, T10);
					T15 = simd_vsubs(T13,(KP250000000)* T14);
					T16 = simd_vadds(T11, T15);
					T3x = simd_vadds(T13, T14);
					T2E = simd_vsubs(T15, T11);
				}
			}
		    {
				floatv4 T1s, T1l, T1o, T1p, T1e, T1h, T1i, T1r;
				
				simd_load(T1r,(float *)&ri[rs*2 + i*ms]);	
				
				Trr = simd_vshff(Tw,Tw,MM_SHUFFLE(2,2,0,0));
				Tii = simd_vshff(Tw,Tw,MM_SHUFFLE(3,3,1,1));
				Ttemp = simd_vmuls(Trr,T1r);
				Tim = simd_vshff(T1r,T1r,MM_SHUFFLE(2,3,0,1));
				Ttemp1 = simd_vmuls(Tim,Tne);
				T1s = simd_vnmas(Tii,Ttemp1,Ttemp);
			
				{
					floatv4 T1k, T1n, T1d, T1g;
					
					simd_load(T1k,(float *)&ri[rs*12 + i*ms]);
					simd_load(T1n,(float *)&ri[rs*17 + i*ms]);
					simd_load(T1d,(float *)&ri[rs*7+ i*ms]);
					simd_load(T1g,(float *)&ri[rs*22 + i*ms]);	
					
					Trr = simd_vshff(T1j,T1j,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1j,T1j,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1k);
					Tim = simd_vshff(T1k,T1k,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1l = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(T1m,T1m,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1m,T1m,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1n);
					Tim = simd_vshff(T1n,T1n,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1o = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					
					T1p = simd_vadds(T1l, T1o);
					
					Trr = simd_vshff(T1c,T1c,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1c,T1c,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1d);
					Tim = simd_vshff(T1d,T1d,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1e = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					Trr = simd_vshff(T1f,T1f,MM_SHUFFLE(2,2,0,0));
					Tii = simd_vshff(T1f,T1f,MM_SHUFFLE(3,3,1,1));
					Ttemp = simd_vmuls(Trr,T1g);
					Tim = simd_vshff(T1g,T1g,MM_SHUFFLE(2,3,0,1));
					Ttemp1 = simd_vmuls(Tim,Tne);
					T1h = simd_vnmas(Tii,Ttemp1,Ttemp);
					
					T1i = simd_vadds(T1e, T1h);
				}
				{
					floatv4 T1w, T1x, T1q, T1t, T1u;
					T1w = simd_vsubs(T1e, T1h);
					T1x = simd_vsubs(T1l, T1o);
					T1y = simd_vadds((KP293892626)* T1x,(KP475528258)* T1w);
					T2N = simd_vsubs((KP475528258)* T1x,(KP293892626)* T1w);
					T1q = (KP559016994)* simd_vsubs(T1i, T1p);
					T1t = simd_vadds(T1i, T1p);
					T1u = simd_vsubs(T1s,(KP250000000)* T1t);
					T1v = simd_vadds(T1q, T1u);
					T3z = simd_vadds(T1s, T1t);
					T2O = simd_vsubs(T1u, T1q);
				}
		    }
		    {
				floatv4 T3J, T3K, T3D, T3E, T3C, T3F, T3L, T3G;
				{
					floatv4 T3H, T3I, T3y, T3B;
					T3H = simd_vsubs(T3w, T3x);
					T3I = simd_vsubs(T3z, T3A);
					T3J = simd_vadds((KP587785252)* T3I,(KP951056516)* T3H);
					Tim = simd_vshff(T3J,T3J,MM_SHUFFLE(2,3,0,1));
					T3J = simd_vmuls(Tim,Tne);
					
					T3K = simd_vsubs((KP951056516)* T3I,(KP587785252)* T3H);
					Tim = simd_vshff(T3K,T3K,MM_SHUFFLE(2,3,0,1));
					T3K = simd_vmuls(Tim,Tne);
					
					T3D = simd_vadds(Tl, Tm);
					
					T3y = simd_vadds(T3w, T3x);
					T3B = simd_vadds(T3z, T3A);
					T3E = simd_vadds(T3y, T3B);
					T3C = (KP559016994)* simd_vsubs(T3y, T3B);
					T3F = simd_vsubs(T3D,(KP250000000)* T3E);
					
				}
				
				T3L = simd_vsubs(T3F, T3C);
				T3G = simd_vadds(T3C, T3F);

				simd_store(simd_vadds(T3D, T3E),(float *)&ri[rs*0+ i*ms]);
				simd_store(simd_vadds(T3K, T3L),(float *)&ri[rs*10 + i*ms]);	
				simd_store(simd_vsubs(T3L, T3K),(float *)&ri[rs*15+ i*ms]);
				simd_store(simd_vsubs(T3G, T3J),(float *)&ri[rs*5+ i*ms]);	
				simd_store(simd_vadds(T3J, T3G),(float *)&ri[rs*20+ i*ms]);		
			
			
		    }
		    {
				floatv4 To, T2n, T2o, T2p, T2x, T2y, T2z, T2u, T2v, T2w, T2q, T2r, T2s, T29, T2i;
				floatv4 T2e, T2g, T1Y, T2j, T2b, T2c, T2B, T2C;
				To = simd_vadds(Tk, Tn);
				T2n = simd_vadds((KP535826794)* TK,(KP1_688655851)* TN);
				T2o = simd_vadds((KP637423989)* T16,(KP1_541026485)* T19);
				T2p = simd_vsubs(T2n, T2o);
				T2x = simd_vadds((KP904827052)* T1v,(KP851558583)* T1y);
				T2y = simd_vadds((KP125333233)* T1S,(KP1_984229402)* T1V);
				T2z = simd_vadds(T2x, T2y);
				T2u = simd_vsubs((KP1_071653589)* TN,(KP844327925)* TK);
				T2v = simd_vsubs((KP770513242)* T16,(KP1_274847979)* T19);
				T2w = simd_vadds(T2u, T2v);
				T2q = simd_vsubs((KP1_809654104)* T1y,(KP425779291)* T1v);
				T2r = simd_vsubs((KP250666467)* T1V,(KP992114701)* T1S);
				T2s = simd_vadds(T2q, T2r);
				{
					floatv4 T23, T24, T25, T26, T27, T28;
					T23 = simd_vadds((KP248689887)* TK,(KP1_937166322)* TN);
					T24 = simd_vadds((KP844327925)* T16,(KP1_071653589)* T19);
					T25 = simd_vadds(T23, T24);
					T26 = simd_vadds((KP481753674)* T1v,(KP1_752613360)* T1y);
					T27 = simd_vadds((KP684547105)* T1S,(KP1_457937254)* T1V);
					T28 = simd_vadds(T26, T27);
					T29 = simd_vadds(T25, T28);
					T2i = simd_vsubs(T27, T26);
					T2e = (KP559016994)* simd_vsubs(T28, T25);
					T2g = simd_vsubs(T24, T23);
				}
				{
					floatv4 TO, T1a, T1b, T1z, T1W, T1X;
					TO = simd_vsubs((KP968583161)* TK,(KP497379774)* TN);
					T1a = simd_vsubs((KP535826794)* T16,(KP1_688655851)* T19);
					T1b = simd_vadds(TO, T1a);
					T1z = simd_vsubs((KP876306680)* T1v,(KP963507348)* T1y);
					T1W = simd_vsubs((KP728968627)* T1S,(KP1_369094211)* T1V);
					T1X = simd_vadds(T1z, T1W);
					T1Y = simd_vadds(T1b, T1X);
					T2j = (KP559016994)* simd_vsubs(T1b, T1X);
					T2b = simd_vsubs(T1a, TO);
					T2c = simd_vsubs(T1z, T1W);
				}
				{
					floatv4 T1Z, T2a, T2t, T2A;
					T1Z = simd_vadds(To, T1Y);
					T2a = simd_vadds(T22, T29);
					Tim = simd_vshff(T2a,T2a,MM_SHUFFLE(2,3,0,1));
					T2a = simd_vmuls(Tim,Tne);
					
			    
					T2t = simd_vadds(To, simd_vadds(T2p, T2s));
					T2A = simd_vadds(T22, simd_vsubs(T2w, T2z));
					Tim = simd_vshff(T2A,T2A,MM_SHUFFLE(2,3,0,1));
					T2A = simd_vmuls(Tim,Tne);
				  	
					simd_store(simd_vsubs(T1Z, T2a),(float *)&ri[rs*1+ i*ms]);
					simd_store(simd_vadds(T1Z, T2a),(float *)&ri[rs*24 + i*ms]);	
					simd_store(simd_vsubs(T2t, T2A),(float *)&ri[rs*21+ i*ms]);
					simd_store(simd_vadds(T2t, T2A),(float *)&ri[rs*4+ i*ms]);	
				
					
				}

				T2B = simd_vadds(T22, simd_vadds((KP309016994)* T2w, simd_vadds((KP587785252)* simd_vsubs(T2r, T2q), simd_vsubs((KP809016994)* T2z,(KP951056516)* simd_vadds(T2n, T2o)))));
				Tim = simd_vshff(T2B,T2B,MM_SHUFFLE(2,3,0,1));
				T2B = simd_vmuls(Tim,Tne);
				
				T2C = simd_vadds((KP309016994)* T2p, simd_vadds((KP951056516)* simd_vsubs(T2u, T2v), simd_vadds(simd_vsubs(To,(KP809016994)* T2s),(KP587785252)* simd_vsubs(T2y, T2x))));
			 	
				simd_store(simd_vadds(T2B, T2C),(float *)&ri[rs*9+ i*ms]);
				simd_store(simd_vsubs(T2C, T2B),(float *)&ri[rs*16 + i*ms]);	
				
			 
				{
					floatv4 T2f, T2l, T2k, T2m, T2d, T2h;
					T2d = simd_vsubs((KP250000000)* T29, T22);
					T2f = simd_vadds(simd_vadds((KP587785252)* T2b, (KP951056516)* T2c), simd_vsubs(T2d, T2e));
					Tim = simd_vshff(T2f,T2f,MM_SHUFFLE(2,3,0,1));
					T2f = simd_vmuls(Tim,Tne);
				
					T2l = simd_vadds(simd_vsubs((KP951056516)* T2b,(KP587785252)* T2c), simd_vadds(T2d, T2e));
					Tim = simd_vshff(T2l,T2l,MM_SHUFFLE(2,3,0,1));
					T2l = simd_vmuls(Tim,Tne);

					T2h = simd_vsubs(To,(KP250000000)* T1Y);
					T2k = simd_vadds((KP587785252)* T2g, simd_vsubs(simd_vsubs(T2h, T2j),(KP951056516)* T2i));
					T2m = simd_vadds((KP951056516)* T2g, simd_vadds(T2j, simd_vadds(T2h,(KP587785252)* T2i)));

					simd_store(simd_vadds(T2f, T2k),(float *)&ri[rs*11+ i*ms]);
					simd_store(simd_vsubs(T2m, T2l),(float *)&ri[rs*19 + i*ms]);	
					simd_store(simd_vsubs(T2k, T2f),(float *)&ri[rs*14+ i*ms]);
					simd_store(simd_vadds(T2l, T2m),(float *)&ri[rs*6+ i*ms]);
				
				}
		    }
		    {
				floatv4 T2S, T2U, T2F, T2I, T2J, T2Y, T2Z, T30, T2M, T2P, T2Q, T2V, T2W, T2X, T3a;
				floatv4 T3l, T3b, T3k, T3f, T3p, T3i, T3o, T32, T33;
				T2S = simd_vsubs(T2R,(KP587785252)* T20);
				T2U = simd_vsubs(Tn, Tk);
				T2F = simd_vsubs((KP1_984229402)* T2D,(KP125333233)* T2E);
				T2I = simd_vadds((KP684547105)* T2H,(KP1_457937254)* T2G);
				T2J = simd_vsubs(T2F, T2I);
				T2Y = simd_vsubs((KP062790519)* T2O,(KP1_996053456)* T2N);
				T2Z = simd_vadds((KP637423989)* T2L,(KP1_541026485)* T2K);
				T30 = simd_vsubs(T2Y, T2Z);
				T2M = simd_vsubs((KP1_274847979)* T2K,(KP770513242)* T2L);
				T2P = simd_vadds((KP998026728)* T2O,(KP125581039)* T2N);
				T2Q = simd_vsubs(T2M, T2P);
				T2V = simd_vsubs((KP728968627)* T2H,(KP1_369094211)* T2G);
				T2W = simd_vadds((KP992114701)* T2E,(KP250666467)* T2D);
				T2X = simd_vsubs(T2V, T2W);
				{
					floatv4 T34, T35, T36, T37, T38, T39;
					T34 = simd_vsubs((KP1_752613360)* T2G,(KP481753674)* T2H);
					T35 = simd_vadds((KP904827052)* T2E,(KP851558583)* T2D);
					T36 = simd_vsubs(T34, T35);
					T37 = simd_vsubs((KP1_071653589)* T2N,(KP844327925)* T2O);
					T38 = simd_vsubs((KP125581039)* T2K,(KP998026728)* T2L);
					T39 = simd_vadds(T37, T38);
					T3a = (KP559016994)* simd_vsubs(T36, T39);
					T3l = simd_vsubs(T37, T38);
					T3b = simd_vadds(T36, T39);
					T3k = simd_vadds(T34, T35);
				}
				{
					floatv4 T3d, T3e, T3m, T3g, T3h, T3n;
					T3d = simd_vsubs((KP1_809654104)* T2D,(KP425779291)* T2E);
					T3e = simd_vadds((KP876306680)* T2H,(KP963507348)* T2G);
					T3m = simd_vadds(T3e, T3d);
					T3g = simd_vadds((KP535826794)* T2O,(KP1_688655851)* T2N);
					T3h = simd_vadds((KP062790519)* T2L,(KP1_996053456)* T2K);
					T3n = simd_vadds(T3g, T3h);
					T3f = simd_vsubs(T3d, T3e);
					T3p = simd_vadds(T3m, T3n);
					T3i = simd_vsubs(T3g, T3h);
					T3o = (KP559016994)* simd_vsubs(T3m, T3n);
				}
				{
					floatv4 T3u, T3v, T2T, T31;
					T3u = simd_vadds(T2S, T3b);
					Tim = simd_vshff(T3u,T3u,MM_SHUFFLE(2,3,0,1));
					T3u = simd_vmuls(Tim,Tne);

					T3v = simd_vadds(T2U, T3p);

					T2T = simd_vsubs(simd_vadds(T2J, T2Q), T2S);
					Tim = simd_vshff(T2T,T2T,MM_SHUFFLE(2,3,0,1));
					T2T = simd_vmuls(Tim,Tne);
					T31 = simd_vadds(T2U, simd_vadds(T2X, T30));

					simd_store(simd_vadds(T3u, T3v),(float *)&ri[rs*2+ i*ms]);
					simd_store(simd_vsubs(T3v, T3u),(float *)&ri[rs*23 + i*ms]);	
					simd_store(simd_vadds(T2T, T31),(float *)&ri[rs*3+ i*ms]);
					simd_store(simd_vsubs(T31, T2T),(float *)&ri[rs*22+ i*ms]);	
					
				}
				T32 = simd_vadds((KP309016994)* T2X, simd_vsubs(simd_vsubs(simd_vsubs(T2U,(KP951056516)* simd_vadds(T2I, T2F)),(KP587785252)* simd_vadds(T2P, T2M)),(KP809016994)* T30));
				T33 = simd_vsubs(simd_vsubs(simd_vsubs(simd_vsubs((KP309016994)* T2J,(KP951056516)* simd_vadds(T2V, T2W)),(KP809016994)* T2Q),(KP587785252)* simd_vadds(T2Y, T2Z)), T2S);
				Tim = simd_vshff(T33,T33,MM_SHUFFLE(2,3,0,1));
				T33 = simd_vmuls(Tim,Tne);
			
				simd_store(simd_vsubs(T32, T33),(float *)&ri[rs*17+ i*ms]);
				simd_store(simd_vadds(T32, T33),(float *)&ri[rs*8 + i*ms]);
				
				{
					floatv4 T3j, T3s, T3r, T3t, T3c, T3q;
					T3c = simd_vsubs(T2S,(KP250000000)* T3b);
					T3j = simd_vadds(T3a, simd_vadds(T3c, simd_vsubs((KP951056516)* T3f,(KP587785252)* T3i)));
					Tim = simd_vshff(T3j,T3j,MM_SHUFFLE(2,3,0,1));
					T3j = simd_vmuls(Tim,Tne);

					T3s = simd_vadds(T3c, simd_vsubs(simd_vadds((KP587785252)* T3f, (KP951056516)* T3i), T3a));
					Tim = simd_vshff(T3s,T3s,MM_SHUFFLE(2,3,0,1));
					T3s = simd_vmuls(Tim,Tne);

					T3q = simd_vsubs(T2U,(KP250000000)* T3p);
					T3r = simd_vadds((KP951056516)* T3k, simd_vadds(simd_vadds(T3o, T3q),(KP587785252)* T3l));
					T3t = simd_vadds((KP587785252)* T3k, simd_vsubs(simd_vsubs(T3q,(KP951056516)* T3l), T3o));

					simd_store(simd_vadds(T3j, T3r),(float *)&ri[rs*7+ i*ms]);
					simd_store(simd_vsubs(T3t, T3s),(float *)&ri[rs*13 + i*ms]);	
					simd_store(simd_vsubs(T3r, T3j),(float *)&ri[rs*18+ i*ms]);
					simd_store(simd_vadds(T3s, T3t),(float *)&ri[rs*12+ i*ms]);	
					
						
				}
		    }
	    }
	}
	return;

}

#if 0
static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 3),
     VTW(0, 9),
     VTW(0, 24),
     {TW_NEXT, VL, 0}
};
#endif


