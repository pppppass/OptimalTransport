all : generate_all test_all

generate_all : font.py utils.py solver_mosek.py solver_gurobi.py first_ADMM_primal.py first_ADMM_dual.py first_ALM_primal.py first_grad_primal.py discrete_transportation_simplex.py discrete_transportation_simplex_networkx.py

test_all : font_test utils_test solver_mosek_test solver_gurobi_test first_ADMM_primal_test first_ADMM_dual_test first_ALM_primal_test first_grad_primal_test discrete_transportation_simplex_test discrete_transportation_simplex_networkx_test overall_test

convert.py : convert.ipynb
	jupyter nbconvert --execute convert.ipynb

font.py : convert.py
	./convert.py font.ipynb

font_test : font.py
	./font_test.py

utils.py : utils.ipynb font.py
	./convert.py utils.ipynb

utils_test : utils.py
	./utils_test.py

solver_mosek.py : solver_mosek.ipynb utils.py
	./convert.py solver_mosek.ipynb

solver_mosek_test : solver_mosek.py
	./solver_mosek_test.py

solver_gurobi.py : solver_gurobi.ipynb utils.py
	./convert.py solver_gurobi.ipynb

solver_gurobi_test : solver_gurobi.py
	./solver_gurobi_test.py

first_ADMM_primal.py : first_ADMM_primal.ipynb utils.py solver_mosek.py 
	./convert.py first_ADMM_primal.ipynb

first_ADMM_primal_test : first_ADMM_primal.py
	./first_ADMM_primal_test.py

first_ADMM_dual.py : first_ADMM_dual.ipynb utils.py solver_mosek.py
	./convert.py first_ADMM_dual.ipynb

first_ADMM_dual_test : first_ADMM_dual.py
	./first_ADMM_dual_test.py

first_ALM_primal.py : first_ALM_primal.ipynb utils.py solver_mosek.py
	./convert.py first_ALM_primal.ipynb

first_ALM_primal_test : first_ALM_primal.py
	./first_ALM_primal_test.py

first_grad_primal.py : first_grad_primal.ipynb utils.py solver_mosek.py first_ADMM_primal.py
	./convert.py first_grad_primal.ipynb

first_grad_primal_test : first_grad_primal.py
	./first_grad_primal_test.py

discrete_transportation_simplex.py : discrete_transportation_simplex.ipynb utils.py solver_mosek.py
	./convert.py discrete_transportation_simplex.ipynb

discrete_transportation_simplex_test : discrete_transportation_simplex.py
	./discrete_transportation_simplex_test.py

discrete_transportation_simplex_networkx.py : discrete_transportation_simplex_networkx.ipynb utils.py solver_mosek.py
	./convert.py discrete_transportation_simplex_networkx.ipynb

discrete_transportation_simplex_networkx_test : discrete_transportation_simplex_networkx.py
	./discrete_transportation_simplex_networkx_test.py

overall_test : overall.ipynb utils.py first_ADMM_primal.py first_ADMM_dual.py first_ALM_primal.py first_grad_primal.py discrete_transportation_simplex.py discrete_transportation_simplex_networkx.py
	./convert.py overall.ipynb
	./overall_test.py

clean_res :
	rm *.pgf
	rm *.png
	rm *.log

clean_py :
	rm *.py

clean : clean_res clean_py
