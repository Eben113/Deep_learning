def get_img_sect(layer,r_from,r_to,c_from,c_to):
	sub_sect = layer[:,r_from:r_to,c_from:c_to]
	return sub_sect.reshape(-1,1,r_to-r_from,c_to-c_from)
	