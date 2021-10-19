//parabola tests
module openscad_paraboloid (y=10, f=5, rfa=0, fc=1, detail=44){
	// y = height of paraboloid
	// f = focus distance 
	// fc : 1 = center paraboloid in focus point(x=0, y=f); 0 = center paraboloid on top (x=0, y=0)
	// rfa = radius of the focus area : 0 = point focus
	// detail = $fn of cone

	hi = (y+2*f)/sqrt(2);								// height and radius of the cone -> alpha = 45° -> sin(45°)=1/sqrt(2)
	x =2*f*sqrt(y/f);									// x  = half size of parabola
	
   translate([0,0,-f*fc])								// center on focus 
	rotate_extrude(convexity = 10,$fn=detail )		// extrude paraboild
	translate([rfa,0,0])								// translate for fokus area	 
	difference(){
		union(){											// adding square for focal area
			projection(cut = true)																			// reduce from 3D cone to 2D parabola
				translate([0,0,f*2]) rotate([45,0,0])													// rotate cone 45° and translate for cutting
				translate([0,0,-hi/2])cylinder(h= hi, r1=hi, r2=0, center=true, $fn=detail);   	// center cone on tip
			translate([-(rfa+x ),0]) square ([rfa+x , y ]);											// focal area square
		}
		translate([-(2*rfa+x ), -1/2]) square ([rfa+x ,y +1] ); 					// cut of half at rotation center 
	}
}

translate([0,0,0]){
   difference(){
openscad_paraboloid (y=0.07239,f=0.146,rfa= 0,fc=0,detail=120);
translate([0,0,0.005])openscad_paraboloid (y=0.07239,f=0.146,rfa= 0,fc=0,detail=120);
}
}
//detail=200;
//rfa=0;
//f=0.146;
//y=0.07239;
//hi = (y+2*f)/sqrt(2);	
////hi=0.2;
//fc=0;
//x =2*f*sqrt(y/f);
//   translate([0,0,-f*fc])								// center on focus 
//	rotate_extrude(convexity = 10,$fn=detail )		// extrude paraboild
//	translate([rfa,0,0])								// translate for fokus area	 
//	difference(){
//		union(){											// adding square for focal area
//			projection(cut = true)																			// reduce from 3D cone to 2D parabola
//				translate([0,0,f*2]) rotate([45,0,0])													// rotate cone 45° and translate for cutting
//				translate([0,0,-hi/2])cylinder(h= hi, r1=hi, r2=0, center=true, $fn=detail);   	// center cone on tip
//			translate([-(rfa+x ),0]) square ([rfa+x , y ]);											// focal area square
//		}
//		translate([-(2*rfa+x ), -1/2]) square ([rfa+x ,y +1] ); 					// cut of half at rotation center 
//	}
