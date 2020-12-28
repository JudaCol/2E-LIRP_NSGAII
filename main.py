# Archivo main para la ejecucion del algoritmo genetico
# from functions import *
from operators_ga import *
from tqdm import tqdm
import time

# parametros de entrada iniciales
n_clientes = 7  # numero de clientes
n_productos = 3  # numero de productos
n_periodos = 3  # numero de periodos
n_vehiculos_p = 6  # numero de vehiculos de primer nivel
n_vehiculos_s = 9  # numero de vehiculos de segundo nivel
n_centrosregionales = 4  # numero de centros regionales
n_centroslocales = 7  # numero de centros locales
n_poblacion = 50  # numero de inidividuos a generar
n_generaciones = 10
prob_mut = 0.1
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
    costo_f1 = np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1])
    valores_f1.append(costo_f1)
    # costos f2
    cost_sufr_hum = fitness_f2(rutas_segundo_lv, n_periodos, costo_humano, n_centroslocales)
    costo_f2 = -cost_sufr_hum
    valores_f2.append(costo_f2)

# calculo de frentes y distancias
dominancias, distancias_t, frentes_dict = frentes(n_poblacion, valores_f1, valores_f2)
# aplicacion de los operadores en cada generacion
Q_poblation = final_inventarioQ
I_poblation = final_inventarioI
f1_poblation = valores_f1
f2_poblation = valores_f2

print("Ejecutando algoritmo genetico...")
for _ in tqdm(range(n_generaciones)):
    # inicio operadores geneticos
    # seleccion de los padres
    idx_parents = selection_padres(n_poblacion, dominancias, distancias_t)
    # cruce
    p_crossed, hijos, demand_cr_hijos, demand_cl_hijos, Q_hijos, I_hijos, f1_hijos, f2_hijos = crossover(individuos, idx_parents, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demand_cl_poblation, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano)
    # mutacion
    hijos, demandas_cr_hijos, Q_hijos, I_hijos, f1_hijos, f2_hijos = mutation(hijos, demand_cr_hijos, n_centrosregionales, capacidad_cr, n_periodos, n_productos, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, n_centroslocales, costo_vehiculos_s, costo_vehiculos_p, costo_humano, Q_hijos, I_hijos, f1_hijos, f2_hijos, prob_mut)    # consolidacion de la nueva poblacion
    # actualizacion del orden de los parametros de los padres
    demand_cr_poblation_o = []
    demand_cl_poblation_o = []
    Q_poblation_o = []
    I_poblation_o = []
    f1_poblation_o = []
    f2_poblation_o = []
    for padre in idx_parents:
        demand_cr_poblation_o.append(demand_cr_poblation[padre])
        demand_cl_poblation_o.append(demand_cl_poblation[padre])
        Q_poblation_o.append(Q_poblation[padre])
        I_poblation_o.append(I_poblation[padre])
        f1_poblation_o.append(f1_poblation[padre])
        f2_poblation_o.append(f2_poblation[padre])
    big_poblation = p_crossed + hijos
    demand_cr_big_poblation = demand_cr_poblation_o + demandas_cr_hijos
    demand_cl_big_poblation = demand_cl_poblation_o + demand_cl_hijos
    Q_big_poblation = Q_poblation_o + Q_hijos
    I_big_poblation = I_poblation_o + I_hijos
    fitness1_big_poblation = f1_poblation_o + f1_hijos
    fitness2_big_poblation = f2_poblation_o + f2_hijos
    # construccion de los frentes de la nueva poblacion
    dominancias, distancias_t, frentes_dict = frentes(len(big_poblation), fitness1_big_poblation, fitness2_big_poblation)
    # seleccion natural para reducir el tamaño a la problacion original
    idx_poblation = reduction(n_poblacion, frentes_dict, distancias_t)
    individuos = []
    demand_cr_poblation = []
    demand_cl_poblation = []
    Q_poblation = []
    I_poblation = []
    f1_poblation = []
    f2_poblation = []
    for idx_p in idx_poblation:
        individuos.append(big_poblation[idx_p])
        demand_cr_poblation.append(demand_cr_big_poblation[idx_p])
        demand_cl_poblation.append(demand_cl_big_poblation[idx_p])
        Q_poblation.append(Q_big_poblation[idx_p])
        I_poblation.append(I_big_poblation[idx_p])
        f1_poblation.append(fitness1_big_poblation[idx_p])
        f2_poblation.append(fitness2_big_poblation[idx_p])
    # calculo de los frentes y distancias de la nueva generacion
    dominancias, distancias_t, frentes_dict = frentes(n_poblacion, f1_poblation, f2_poblation)
TimeEnd = time.time()
print("El tiempo de ejecucion del algoritmo es de {} segundos".format(TimeEnd-TimeStart))
# extraccion de los mejores individuos
print("individuo                     f1                                f2")
for id_bob in range(len(frentes_dict[0])):
    print("{0:3d}                    {1:.4f}                      {2:.4f}".format(id_bob, f1_poblation[id_bob], f2_poblation[id_bob]))
