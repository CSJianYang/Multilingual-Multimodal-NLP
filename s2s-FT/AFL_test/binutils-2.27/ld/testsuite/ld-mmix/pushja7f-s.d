#source: start.s
#source: pushja.s
#source: a.s
#as: -x
#ld: -m mmo
#objdump: -dr

# Like pushja7f, but with PUSHJ stub.

.*:     file format mmo
Disassembly of section \.text:
0+ <(Main|_start)>:
   0:	e3fd0001 	setl \$253,0x1
0+4 <pushja>:
   4:	e3fd0002 	setl \$253,0x2
   8:	f20c0002 	pushj \$12,10 <a>
   c:	e3fd0003 	setl \$253,0x3
0+10 <a>:
  10:	e3fd0004 	setl \$253,0x4
