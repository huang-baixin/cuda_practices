APPS=demo simpleShfl reduceInteger 
# APPS=simpleShfl

all : ${APPS}

% : %.cu
	# nvcc -o $@ $<
	nvcc -o $@ $<
clean :
	rm -f ${APP}




