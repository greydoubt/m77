	IDEAL
	JUMPS
	LOCALS
	P286N

	DOSSEG
	MODEL	LARGE
;à®æ¥¤ãàë ¨ äã­ªæ¨¨
        PUBLIC  _jstat	;áâ âãá ¤¦®©áâ¨ª®¢

;¥à¥¬¥­­ë¥
	PUBLIC	_a_1
	PUBLIC  _a_2
	PUBLIC  _b_1
	PUBLIC  _b_2
	PUBLIC  _xa
	PUBLIC  _ya
	PUBLIC  _xb
	PUBLIC  _yb

;ðððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððð
SEGMENT	CODE	WORD
	ASSUME	cs:CODE

_a_1	db	(?)
_a_2	db      (?)
_b_1	db      (?)
_b_2	db      (?)


_xa	dw	(?)
_ya	dw	(?)
_xb	dw	(?)
_yb	dw      (?)


	ENDS

ðððððððððððððððððððððððððððððððððððððððððððððððððð

SEGMENT	CODE	WORD
	ASSUME	cs:CODE

;	jstat(void)

	PROC	_jstat FAR

        push    bp
	mov	bp,sp

	PUSH DX

	;mov	AH,84h
	;mov	DX,1

	mov	CX,0	;®¡­ã«¨¬ ¯®«®¦¥­¨ï àãçª¨
	mov	[_xa],CX
	mov	[_ya],CX
	mov	[_xb],CX
	mov	[_yb],CX

	;cli
	;int	15h
	mov	dx,201h
	out	dx,al
;	mov	cx,-1
@@gain:
	in	al,dx
	inc	cx

	test	al,1
	jnz	@@g1

	cmp	[_xa],0
	jnz	@@g1
	mov	[_xa],CX
@@g1:
	test	al,2
	jnz	@@g2
	cmp	[_ya],0
	jnz	@@g2
	mov	[_ya],CX
@@g2:
	test	al,4
	jnz	@@g3
	cmp	[_xb],0
	jnz	@@g3
	mov	[_xb],CX
@@g3:
	test	al,8
	jnz	@@g4
	cmp	[_yb],0
	jnz	@@g4
	mov	[_yb],CX
@@g4:
	cmp	CX,1000h
	jne	@@gain
	;sti

	mov	DX,201h
	out	DX,AL
	in	AL,DX

	mov	bl,al
	and	bl,16
	mov	[_a_1],bl
	mov	bl,al
	and	bl,32
	mov	[_a_2],bl
	mov	bl,al
	and	bl,64
	mov	[_b_1],bl
	mov	bl,al
	and	bl,128
	mov	[_b_2],bl

	pop	DX
	mov	sp,bp
	pop	bp
	ret
	ENDP

	ENDS
;ðððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððð

	END



