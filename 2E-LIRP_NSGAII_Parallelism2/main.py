# Archivo main para la ejecucion del algoritmo genetico
# from functions import *
import threading as thr
import multiprocessing as mp
from operators_ga import *
import time
import matplotlib.pyplot as plt

# parametros de entrada iniciales
n_clientes = 140  # numero de clientes
n_productos = 2  # numero de productos
n_periodos = 2  # numero de periodos
n_vehiculos_p = 11  # numero de vehiculos de primer nivel
n_vehiculos_s = 12  # numero de vehiculos de segundo nivel
n_centrosregionales = 3  # numero de centros regionales
n_centroslocales = 5  # numero de centros locales
n_poblacion = 100  # numero de inidividuos a generar
prob_mut = 0.1
n_generaciones = 10
individuos = []
demand_cr_poblation = []
demand_cl_poblation = []
final_inventarioQ = []
final_inventarioI = []
valores_f1 = []
valores_f2 = []


TimeStart = time.time()
# obtencion de las demandas y capacidades dadas en matrices en un archivo de excel
demanda_clientes, capacidad_vehiculos_p, capacidad_vehiculos_s, capacidad_cr, capacidad_cl, inventario, costo_inventario, costo_instalaciones_cr, costo_instalaciones_cl, costo_vehiculos_p, costo_vehiculos_s, costo_compraproductos, costo_transporte, costo_rutas_p, costo_rutas_s, costo_humano = read_data(n_clientes, n_productos, n_periodos, n_vehiculos_p, n_vehiculos_s, n_centrosregionales, n_centroslocales)

print("Generando poblacion inicial...")
# generacion de la poblacion inicial
for i in tqdm(range(n_poblacion)):
    # Generacion de un individuo
    asignaciones_primer_lv, asignaciones_segundo_lv, rutas_primer_lv, rutas_segundo_lv, demandas_cr_full, demandas_cl = individuo(n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cl, capacidad_cr, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes)
    # almacenamiento del individuo en una lista
    individuos.append([asignaciones_primer_lv, rutas_primer_lv, asignaciones_segundo_lv, rutas_segundo_lv])
    # almacenamiento de las demandas de centros regionales en una lista
    demand_cr_poblation.append(demandas_cr_full)
    # almacenamiento de las demandas de centros locales en una lista
    demand_cl_poblation.append(demandas_cl)
    # Generacion de los valores Q e I de la gestion de inventarios
    valoresQ, valoresI = fun_inventario(demandas_cr_full, n_periodos, n_productos, n_centrosregionales, capacidad_cr, inventario)
    final_inventarioQ.append(valoresQ)
    final_inventarioI.append(valoresI)
    # costos f1
    cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1 = fitness_f1(n_periodos, n_productos, asignaciones_segundo_lv, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, valoresQ, valoresI, rutas_segundo_lv, rutas_primer_lv, costo_rutas_s, costo_rutas_p, n_centroslocales, n_centrosregionales, costo_vehiculos_s, costo_vehiculos_p)
    o = 10**-3
    costprod = costprod*o
    costtrans = costtrans*o
    costinv = costinv*o
    costo_f1 = round(np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1]), 3)
    valores_f1.append(costo_f1)
    # costos f2
    cost_sufr_hum = round(fitness_f2(rutas_segundo_lv, n_periodos, costo_humano, n_centroslocales), 3)
    costo_f2 = -cost_sufr_hum
    valores_f2.append(costo_f2)

Q_poblation = final_inventarioQ
I_poblation = final_inventarioI
f1_poblation = valores_f1
f2_poblation = valores_f2

# fraccionamiento de la poblacion inicial
individuos1 = individuos[:int(n_poblacion/2)]
demand_cl_poblation1 = demand_cl_poblation[:int(n_poblacion/2)]
demand_cr_poblation1 = demand_cr_poblation[:int(n_poblacion/2)]
Q_poblation1 = Q_poblation[:int(n_poblacion/2)]
I_poblation1 = I_poblation[:int(n_poblacion/2)]
f1_poblation1 = f1_poblation[:int(n_poblacion/2)]
f2_poblation1 = f2_poblation[:int(n_poblacion/2)]
# calculo de frentes y distancias
dominancias1, distancias_t1, frentes_dict1 = frentes(len(individuos1), f1_poblation1, f2_poblation1)


individuos2 = individuos[int(n_poblacion/2):]
demand_cl_poblation2 = demand_cl_poblation[int(n_poblacion/2):]
demand_cr_poblation2 = demand_cr_poblation[int(n_poblacion/2):]
Q_poblation2 = Q_poblation[int(n_poblacion/2):]
I_poblation2 = I_poblation[int(n_poblacion/2):]
f1_poblation2 = f1_poblation[int(n_poblacion/2):]
f2_poblation2 = f2_poblation[int(n_poblacion/2):]
# calculo de frentes y distancias
dominancias2, distancias_t2, frentes_dict2 = frentes(len(individuos2), f1_poblation2, f2_poblation2)


# aplicacion de los operadores en cada generacion por hilos
colector = mp.Array('d', [0, 0])
colector.value = [[], []]
print("Ejecutando algoritmo genetico en hilos...")
t1 = thr.Thread(target=run_ga, args=(individuos1, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demand_cl_poblation1, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano, n_generaciones, dominancias1, distancias_t1, demand_cr_poblation1, Q_poblation1, I_poblation1, f1_poblation1, f2_poblation1, prob_mut, colector, 0))
t2 = thr.Thread(target=run_ga, args=(individuos2, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demand_cl_poblation2, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano, n_generaciones, dominancias2, distancias_t2, demand_cr_poblation2, Q_poblation2, I_poblation2, f1_poblation2, f2_poblation2, prob_mut, colector, 1))
t1.start()
t2.start()
t1.join()
t2.join()
TimeEnd = time.time()
print("El tiempo de ejecucion del algoritmo es de {} segundos".format(TimeEnd-TimeStart))
# extraccion de la mejor aproximacion del frente de pareto de cada isla

ind_pareto_temp = colector.value[0][0] + colector.value[1][0]
q_pareto_temp = colector.value[0][1] + colector.value[1][1]
i_pareto_temp = colector.value[0][2] + colector.value[1][2]
f1_pareto_temp = colector.value[0][3] + colector.value[1][3]
f2_pareto_temp = colector.value[0][4] + colector.value[1][4]

# recalculo de los frentes y distancia de las soluciones
dominancias_pareto, distancias_pareto, frentes_dict_pareto = frentes(len(ind_pareto_temp), f1_pareto_temp, f2_pareto_temp)

# depuracion de individuos repetidos
f_poblation = []
for x in range(len(f1_pareto_temp)):
    f_poblation.append([f1_pareto_temp[x], f2_pareto_temp[x]])
idx_pareto = []
fit_pareto = []
for idx, fit in enumerate(f_poblation):
    if fit not in fit_pareto:
        idx_pareto.append(idx)
        fit_pareto.append(fit)

# almacenaje de las estructuras de los individuos de pareto
indi_pareto = []
q_pob_pareto = []
i_pob_pareto = []
f1_pob_pareto = []
f2_pob_pareto = []
for idx_f, idx_par in enumerate(idx_pareto):
    indi_pareto.append(ind_pareto_temp[idx_par])
    q_pob_pareto.append(q_pareto_temp[idx_par])
    i_pob_pareto.append(i_pareto_temp[idx_par])
    f1_pob_pareto.append(fit_pareto[idx_f][0])
    f2_pob_pareto.append(fit_pareto[idx_f][1])

# almacenaje de las estructuras de los mejores individuos en variables independientes
best_inds = []
best_q = []
best_i = []
best_f1 = []
best_f2 = []
for idx_b in frentes_dict_pareto[0]:
    best_inds.append(indi_pareto[idx_b])
    best_q.append(q_pob_pareto[idx_b])
    best_i.append(i_pob_pareto[idx_b])
    best_f1.append(f1_pob_pareto[idx_b])
    best_f2.append(f2_pob_pareto[idx_b])


# impresion de los individuos y su fitness
print("\n FITNESS DE LOS MEJORES INDIVIDUOS DE LA APROXIMACION DE PARETO - MEJOR FRENTE \n")
print("individuo                  f1                                f2")
for id_p in frentes_dict_pareto[0]:
    print("{0:3d}                    {1:.3f}                      {2:.3f}".format(id_p + 1, f1_pob_pareto[id_p], f2_pob_pareto[id_p]))
print("\n FITNESS DE LOS INDIVIDUOS DE LA APROXIMACION DE PARETO - POBLACION FINAL \n")
print("individuo                  f1                                f2")
for id_bob in range(len(indi_pareto)):
    print("{0:3d}                    {1:.3f}                      {2:.3f}".format(id_bob+1, f1_pob_pareto[id_bob], f2_pob_pareto[id_bob]))

# graficos
# graficos de fitness
# valores del eje x - f1
# valores del eje y - f2
x = []
y = []
f1_pareto_g = {}
for xs in frentes_dict_pareto[0]:
    f1_pareto_g[xs] = f1_pob_pareto[xs]

f1_pareto_o = sorted(f1_pareto_g.items(), key=operator.itemgetter(1), reverse=False)  # ordenamos las distancias de apilamiento de mayor a menor

for xi in f1_pareto_o:
    x.append(xi[1])
    y.append(f2_pob_pareto[xi[0]])

# parametros del grafico
fig, axs = plt.subplots()
# grafico fitness total
axs.plot(x, y, marker='o', ms=5, mec='r', mfc='r', linestyle=':', color='g')
axs.set_title("Frontera de pareto")
axs.set_xlabel("F1")
axs.set_ylabel("F2")
plt.show()
