CASE_B(0xfe)               /* GRP4 Eb */
    {
	    GetRM;Bitu which=(rm>>3)&7;
	    switch (which) {
			case 0x00:     /* INC Eb */
			    RMEb(INCB);
			    break;
			case 0x01:     /* DEC Eb */
			    RMEb(DECB);
			    break;
			case 0x07:     /* CallBack */
			    {
			        Bitu cb=Fetchw();
			        FillFlags();SAVEIP;
			        return cb;
			    }
			default:
				E_Exit("Illegal GRP4 Call %d",(rm>>3) & 7);
				break;
	    }
	    break;
    }
