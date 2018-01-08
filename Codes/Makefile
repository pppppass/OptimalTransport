default: all
.PHONY : default

all: generate_all test_all
.PHONY : all

generate_all: convert.py font.py handler.py dataset.py stats.py solver_mosek.py solver_gurobi.py first_ADMM_primal.py first_ADMM_dual.py first_ALM_primal.py first_grad_primal.py discrete_transportation_simplex.py discrete_transportation_simplex_networkx.py multiscale_image_mosek.py multiscale_image_func.py multiscale_point_mosek.py entropy_ADMM_primal.py entropy_sinkhorn.py
.PHONY : generate_all

test_all: font_test handler_test dataset_test stats_test solver_mosek_test solver_gurobi_test first_ADMM_primal_test first_ADMM_dual_test first_ALM_primal_test first_grad_primal_test discrete_transportation_simplex_test discrete_transportation_simplex_networkx_test multiscale_image_mosek_test multiscale_image_func_test multiscale_point_mosek_test entropy_ADMM_primal_test entropy_sinkhorn_test
.PHONY : test_all

DOTmark_1.0.zip:
	wget http://www.stochastik.math.uni-goettingen.de/DOTmark_1.0.zip
	unzip DOTmark_1.0.zip

convert.py: convert.ipynb
	jupyter nbconvert --execute convert.ipynb

font.py: convert.py
	./convert.py font.ipynb

font_test: font.py
	./font_test.py
.PHONY : font_test

handler.py: handler.ipynb convert.py
	./convert.py handler.ipynb

handler_test: handler.py font.py
	./handler_test.py
.PHONY : handler_test

dataset.py: dataset.ipynb convert.py
	./convert.py dataset.ipynb

dataset_test: dataset.py font.py handler.py DOTmark_1.0.zip
	./dataset_test.py
.PHONY : dataset_test

stats.py: stats.ipynb convert.py
	./convert.py stats.ipynb

stats_test: stats.py font.py handler.py dataset.py
	./stats_test.py

solver_mosek.py: solver_mosek.ipynb convert.py
	./convert.py solver_mosek.ipynb

solver_mosek_test: solver_mosek.py font.py handler.py dataset.py stats.py
	./solver_mosek_test.py

solver_gurobi.py: solver_gurobi.ipynb convert.py
	./convert.py solver_gurobi.ipynb

solver_gurobi_test: solver_gurobi.py font.py handler.py dataset.py stats.py
	./solver_gurobi_test.py

first_ADMM_primal.py: first_ADMM_primal.ipynb convert.py
	./convert.py first_ADMM_primal.ipynb

first_ADMM_primal_test: first_ADMM_primal.py font.py handler.py dataset.py stats.py solver_mosek.py
	./first_ADMM_primal_test.py

first_ADMM_dual.py: first_ADMM_dual.ipynb convert.py
	./convert.py first_ADMM_dual.ipynb

first_ADMM_dual_test: first_ADMM_dual.py font.py handler.py dataset.py stats.py solver_mosek.py
	./first_ADMM_dual_test.py

first_ALM_primal.py: first_ALM_primal.ipynb convert.py
	./convert.py first_ALM_primal.ipynb

first_ALM_primal_test: first_ALM_primal.py font.py handler.py dataset.py stats.py solver_mosek.py
	./first_ALM_primal_test.py

first_grad_primal.py: first_grad_primal.ipynb convert.py
	./convert.py first_grad_primal.ipynb

first_grad_primal_test: first_grad_primal.py font.py handler.py dataset.py stats.py solver_mosek.py
	./first_grad_primal_test.py

discrete_transportation_simplex.py: discrete_transportation_simplex.ipynb convert.py
	./convert.py discrete_transportation_simplex.ipynb

discrete_transportation_simplex_test: discrete_transportation_simplex.py font.py handler.py dataset.py stats.py solver_mosek.py
	./discrete_transportation_simplex_test.py

discrete_transportation_simplex_networkx.py: discrete_transportation_simplex_networkx.ipynb convert.py
	./convert.py discrete_transportation_simplex_networkx.ipynb

discrete_transportation_simplex_networkx_test: discrete_transportation_simplex_networkx.py font.py handler.py dataset.py stats.py solver_mosek.py
	./discrete_transportation_simplex_networkx_test.py

multiscale_image_mosek.py: multiscale_image_mosek.ipynb convert.py
	./convert.py multiscale_image_mosek.ipynb

multiscale_image_mosek_test: multiscale_image_mosek.py font.py handler.py dataset.py stats.py solver_mosek.py
	./multiscale_image_mosek_test.py

multiscale_image_func.py: multiscale_image_func.ipynb convert.py
	./convert.py multiscale_image_func.ipynb

multiscale_image_func_test: multiscale_image_func.py font.py handler.py dataset.py stats.py solver_mosek.py first_ADMM_primal.py
	./multiscale_image_func_test.py

multiscale_point_mosek.py: multiscale_point_mosek.ipynb convert.py
	./convert.py multiscale_point_mosek.ipynb

multiscale_point_mosek_test: multiscale_point_mosek.py font.py handler.py dataset.py stats.py solver_mosek.py
	./multiscale_point_mosek_test.py

entropy_ADMM_primal.py: entropy_ADMM_primal.ipynb convert.py
	./convert.py entropy_ADMM_primal.ipynb

entropy_ADMM_primal_test: entropy_ADMM_primal.py font.py handler.py dataset.py stats.py first_ADMM_primal.py
	./entropy_ADMM_primal_test.py

entropy_sinkhorn.py: entropy_sinkhorn.ipynb convert.py
	./convert.py entropy_sinkhorn.ipynb

entropy_sinkhorn_test: entropy_sinkhorn.py font.py handler.py dataset.py stats.py solver_mosek.py
	./entropy_sinkhorn_test.py

overall_extra.py: overall_extra.ipynb generate_all
	./convert.py overall_extra.ipynb

overall_extra_test: overall_extra.py
	./overall_extra.py

overall_point.py: overall_point.ipynb generate_all
	./convert.py overall_point.ipynb

overall_point_test: overall_point.py
	./overall_point.py

overall_image.py: overall_image.ipynb generate_all
	./convert.py overall_image.ipynb

overall_image_test: overall_image.py
	./overall_image.py

overall_supp.py: overall_supp.ipynb generate_all
	./convert.py overall_supp.ipynb

overall_supp_test: overall_supp.py
	./overall_supp.py

figures.py: figures.ipynb generate_all
	./convert.py figures.ipynb

figures_test: figures.py
	./figures.py

clean_res:
	rm *.pgf
	rm *.png
	rm *.log
.PHONY : clean_res

clean_py:
	rm *.py
.PHONY : clean_py

clean: clean_res clean_py
.PHONY : clean

sync:
	cp  convert.ipynb font.ipynb handler.ipynb dataset.ipynb stats.ipynb solver_mosek.ipynb solver_gurobi.ipynb first_ADMM_primal.ipynb first_ADMM_dual.ipynb first_ALM_primal.ipynb first_grad_primal.ipynb discrete_transportation_simplex.ipynb discrete_transportation_simplex_networkx.ipynb multiscale_image_mosek.ipynb multiscale_image_func.ipynb multiscale_point_mosek.ipynb entropy_ADMM_primal.ipynb entropy_sinkhorn.ipynb overall_point.ipynb overall_image.ipynb overall_extra.ipynb overall_supp.ipynb figures.ipynb Makefile environment.yml ../Codes
.PHONY : sync
