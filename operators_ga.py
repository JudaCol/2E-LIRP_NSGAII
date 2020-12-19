import numpy as np
from functions import *
from itertools import groupby


# operador de seleccion de padres
def selection_padres(poblacion, dominancias, distancias):
    parents = []
    for _ in range(poblacion):
        rand_1 = np.random.randint(poblacion)
        while rand_1 in parents:
            rand_1 = np.random.randint(poblacion)
        rand_2 = np.random.randint(poblacion)
        while (rand_2 in parents) and rand_2 == rand_1:
            rand_2 = np.random.randint(poblacion)
        if dominancias[rand_1] < dominancias[rand_2]:
            parents.append(rand_1)
        elif dominancias[rand_1] > dominancias[rand_2]:
            parents.append(rand_2)
        elif dominancias[rand_1] == dominancias[rand_2]:
            if distancias[rand_1] > distancias[rand_2]:
                parents.append(rand_1)
            else:
                parents.append(rand_2)
    return parents


def rutas2(asignacion_lv, n_vehiculos, capacidad_vehiculos, demanda, periodo, n_productos, escalon, mapeo):
    rutas_lv = []
    vehiculos = list(range(1, n_vehiculos+1))
    dicci_asignacion = dictionarize(asignacion_lv)
    rango = (periodo-1)*n_productos
    capacidad_vehiculos_copy = np.copy(capacidad_vehiculos)
    for centro, asignados in dicci_asignacion.items():
        idx_c = 0
        if len(mapeo) > 0:
            vehiculo_temp = np.random.choice(mapeo)
        else:
            vehiculo_temp = np.random.choice(vehiculos)
        ruta_temp = [int(centro), vehiculo_temp, 0]
        veh_cap = np.copy(capacidad_vehiculos_copy[vehiculo_temp-1, :])
        while idx_c < len(asignados):
            if periodo > 1 and escalon == 1:
                dem_c = demanda[idx_c, 0:n_productos]
            else:
                dem_c = demanda[idx_c, rango:rango+n_productos]
            resta = veh_cap - dem_c
            binvec = np.array([binarize(x) for x in resta])
            if binvec.all():
                ruta_temp.append(asignados[idx_c])
                veh_cap -= dem_c
                idx_c += 1
            else:
                if len(ruta_temp) != 0:
                    if ruta_temp[-1] == 0:
                        ruta_temp.pop(-1)
                        ruta_temp.pop(-1)
                        if len(mapeo) > 1:
                            mapeo.pop(mapeo.index(vehiculo_temp))
                            vehiculos.pop(vehiculos.index(vehiculo_temp))
                            vehiculo_temp = np.random.choice(mapeo)
                        elif len(mapeo) == 1:
                            mapeo.pop(mapeo.index(vehiculo_temp))
                            vehiculos.pop(vehiculos.index(vehiculo_temp))
                            vehiculo_temp = np.random.choice(vehiculos)
                        else:
                            vehiculos.pop(vehiculos.index(vehiculo_temp))
                            vehiculo_temp = np.random.choice(vehiculos)
                        veh_cap = capacidad_vehiculos_copy[vehiculo_temp - 1, :]
                        ruta_temp += [vehiculo_temp, 0]
                    else:
                        ruta_temp.append(0)
                        if len(mapeo) > 1:
                            mapeo.pop(mapeo.index(vehiculo_temp))
                            vehiculos.pop(vehiculos.index(vehiculo_temp))
                            vehiculo_temp = np.random.choice(mapeo)
                        elif len(mapeo) ==1:
                            mapeo.pop(mapeo.index(vehiculo_temp))
                            vehiculos.pop(vehiculos.index(vehiculo_temp))
                            vehiculo_temp = np.random.choice(vehiculos)
                        else:
                            vehiculos.pop(vehiculos.index(vehiculo_temp))
                            vehiculo_temp = np.random.choice(vehiculos)
                        veh_cap = capacidad_vehiculos_copy[vehiculo_temp - 1, :]
                        ruta_temp += [vehiculo_temp, 0]
                else:
                    if len(mapeo) > 1:
                        vehiculo_temp = np.random.choice(mapeo)
                    else:
                        vehiculo_temp = np.random.choice(vehiculos)
                    ruta_temp = [int(centro), vehiculo_temp, 0]
                    veh_cap = np.copy(capacidad_vehiculos_copy[vehiculo_temp - 1, :])
        if len(mapeo) > 0:
            mapeo.pop(mapeo.index(vehiculo_temp))
            vehiculos.pop(vehiculos.index(vehiculo_temp))
        else:
            vehiculos.pop(vehiculos.index(vehiculo_temp))
        ruta_temp.append(0)
        rutas_lv.append(ruta_temp)
    rutas_lv.append(-1)

    return rutas_lv


def extract_c(padre, n_periodos, ext):
    c_extract = []
    for t in range(n_periodos):
        for c in padre[0][t][ext]:
            if c not in c_extract:
                c_extract.append(c)  # centros regionales habilitados en el primer lvl
    return c_extract


def extract_vh(padre, n_periodos, ext):
    rutas_padre = np.copy(padre[ext])
    vh_habs = []
    for t in range(n_periodos):
        ruta_periodo = []
        vh_temp = []
        for ruta in rutas_padre:
            if ruta == -1:
                break
            else:
                ruta_periodo.append(ruta)
        rutas_padre = np.delete(rutas_padre, np.array(range(len(ruta_periodo) + 1)))
        rut_cl = []
        for ruta in ruta_periodo:
            ruta_f = [list(g) for k, g in groupby(ruta, lambda x: x != 0) if k]
            rut_cl.append(ruta_f)
        for visit in rut_cl:
            vh_temp.append(visit[0][1])
            for idx in range(1, len(visit)):
                if idx % 2 == 0:
                    vh_temp.append(visit[idx][0])
        vh_habs.append(vh_temp)
    return vh_habs


def reconstruction_1(n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_p, capacidad_cr, capacidad_vehiculos_p, demandas_cl_cross_seg1, seg2_cr_habs_h1, seg2_vh1_habs, inventario, seg1_escalon, seg1_rutas, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano):
    rec_seg2_escalon_full = []
    rec_seg2_demand_cr_full = []
    rec_seg2_rutas = []
    rec_seg2_demand_cl_full = []
    for periodo_actual in range(1, n_periodos + 1):
        # extraccion de datos de los cl del seg1
        seg1_n_cl_habs, seg1_demanda_cl_np, seg1_cl_habs = maping(demandas_cl_cross_seg1[periodo_actual - 1])
        rec_seg2_escalon, rec_seg2_demand_cr = asignaciones(seg1_n_cl_habs, n_centrosregionales, periodo_actual, n_productos, capacidad_cr, seg1_demanda_cl_np, seg2_cr_habs_h1, 1)
        rec_seg2_escalon[0, :] = seg1_cl_habs
        rec_seg2_rutas += rutas2(rec_seg2_escalon, n_vehiculos_p, capacidad_vehiculos_p, seg1_demanda_cl_np, periodo_actual, n_productos, 1, seg2_vh1_habs[periodo_actual - 1])
        rec_seg2_escalon_full.append(rec_seg2_escalon)
        rec_seg2_demand_cr_dict = dictionarize_cr(rec_seg2_demand_cr)
        rec_seg2_demand_cr_full.append(rec_seg2_demand_cr_dict)
        rec_seg2_demand_cl_full.append(demandas_cl_cross_seg1[periodo_actual - 1])
    rec_seg2_Q, rec_seg2_I = fun_inventario(rec_seg2_demand_cr_full, n_periodos, n_productos, n_centrosregionales, capacidad_cr, inventario)
    # consolidacion del hijo1
    hijo1 = [rec_seg2_escalon_full, rec_seg2_rutas, seg1_escalon, seg1_rutas]
    hijo1_Q = rec_seg2_Q
    hijo1_I = rec_seg2_I
    # fitness para el hijo1
    # costos f1
    cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1 = fitness_f1(n_periodos, n_productos, seg1_escalon, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, hijo1_Q, hijo1_I, seg1_rutas, rec_seg2_rutas, costo_rutas_s, costo_rutas_p, n_centroslocales, n_centrosregionales, costo_vehiculos_s, costo_vehiculos_p)
    o = 10 ** -3
    costprod = costprod * o
    costtrans = costtrans * o
    costinv = costinv * o
    hijo1_costo_f1 = round(np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1]), 3)
    # costos f2
    cost_sufr_hum = round(fitness_f2(seg1_rutas, n_periodos, costo_humano, n_centroslocales), 3)
    hijo1_costo_f2 = -cost_sufr_hum
    return hijo1, rec_seg2_demand_cr_full, rec_seg2_demand_cl_full, hijo1_I, hijo1_Q, hijo1_costo_f1, hijo1_costo_f2


def reconstruction_2(n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cl, capacidad_cr, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demandas_cl_cross_seg1, vh1_habs_h2, seg2_cr_habs_h2, vh2_habs_h2, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano):
    rec_seg2_escalon_full_h2 = []
    rec_seg2_demand_cr_full_h2 = []
    rec_seg2_rutas_h2 = []
    rec_seg1_rutas_h2 = []
    rec_seg1_escalon_full_h2 = []
    rec_seg1_demand_cl_full_h2 = []
    for periodo_actual in range(1, n_periodos + 1):
        # extraccion de datos de los cl del seg1
        seg1_n_cl_habs, seg1_demanda_cl_np, seg1_cl_habs = maping(demandas_cl_cross_seg1[periodo_actual - 1])
        rec_seg1_escalon, rec_seg1_demand_cl = asignaciones(n_clientes, n_centroslocales, periodo_actual, n_productos, capacidad_cl, demanda_clientes, seg1_cl_habs, 2)
        seg1_n_cl_habs, seg1_demanda_cl_np, seg1_cl_habs = maping(rec_seg1_demand_cl)
        rec_seg1_rutas_h2 += rutas2(rec_seg1_escalon, n_vehiculos_s, capacidad_vehiculos_s, demanda_clientes, periodo_actual, n_productos, 2, vh2_habs_h2[periodo_actual - 1])
        rec_seg2_escalon, rec_seg2_demand_cr = asignaciones(seg1_n_cl_habs, n_centrosregionales, periodo_actual, n_productos, capacidad_cr, seg1_demanda_cl_np, seg2_cr_habs_h2, 1)
        rec_seg2_escalon[0, :] = seg1_cl_habs
        rec_seg2_rutas_h2 += rutas2(rec_seg2_escalon, n_vehiculos_p, capacidad_vehiculos_p, seg1_demanda_cl_np, periodo_actual, n_productos, 1, vh1_habs_h2[periodo_actual - 1])
        rec_seg2_escalon_full_h2.append(rec_seg2_escalon)
        rec_seg2_demand_cr_dict_h2 = dictionarize_cr(rec_seg2_demand_cr)
        rec_seg2_demand_cr_full_h2.append(rec_seg2_demand_cr_dict_h2)
        rec_seg1_demand_cl_full_h2.append(rec_seg1_demand_cl)
        rec_seg1_escalon_full_h2.append(rec_seg1_escalon)

    rec_seg2_Q_h2, rec_seg2_I_h2 = fun_inventario(rec_seg2_demand_cr_full_h2, n_periodos, n_productos, n_centrosregionales, capacidad_cr, inventario)
    # consolidacion del hijo 2
    hijo2 = [rec_seg2_escalon_full_h2, rec_seg2_rutas_h2, rec_seg1_escalon_full_h2, rec_seg1_rutas_h2]
    hijo2_Q = rec_seg2_Q_h2
    hijo2_I = rec_seg2_I_h2
    # fitness para el hijo2
    # costos f1
    cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1 = fitness_f1(n_periodos, n_productos, rec_seg1_escalon_full_h2, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, hijo2_Q, hijo2_I, rec_seg1_rutas_h2, rec_seg2_rutas_h2, costo_rutas_s, costo_rutas_p, n_centroslocales, n_centrosregionales, costo_vehiculos_s, costo_vehiculos_p)
    o = 10 ** -3
    costprod = costprod * o
    costtrans = costtrans * o
    costinv = costinv * o
    hijo2_costo_f1 = round(np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1]), 3)
    # costos f2
    cost_sufr_hum = round(fitness_f2(rec_seg1_rutas_h2, n_periodos, costo_humano, n_centroslocales), 3)
    hijo2_costo_f2 = -cost_sufr_hum

    return hijo2, rec_seg2_demand_cr_full_h2, rec_seg1_demand_cl_full_h2, hijo2_I, hijo2_Q, hijo2_costo_f1, hijo2_costo_f2


def crossover(individuos, idx_padres, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demandas_cl_cross, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano):
    crossed = []
    hijos = []
    demand_cr_hijos = []
    demand_cl_hijos = []
    I_hijos = []
    Q_hijos = []
    f1_hijos = []
    f2_hijos = []
    p_crossed = []
    for idx in range(0, len(idx_padres), 2):
        padre1 = individuos[idx]
        padre2 = individuos[idx+1]
        p_crossed.append(padre1)
        p_crossed.append(padre2)
        demandas_cl_cross_seg2 = demandas_cl_cross[idx]  # demandas padre 1
        demandas_cl_cross_seg1 = demandas_cl_cross[idx+1]  # demandas padre 2
        # en el hijo 1 se parcializa el segmento 2
        # elementos que se heredan al hijo 1 del padre 2 y padre 1
        seg1_escalon = padre2[2]  # asignaciones de segundo escalon
        seg1_rutas = padre2[3]  # rutas de segundo escalon
        seg2_cr_habs_h1 = extract_c(padre1, n_periodos, 1)  # centros regionales habilitados en el primer escalon del padre 1
        seg2_vh1_habs = extract_vh(padre1, n_periodos, 1)  # vehiculos de primer lvl habilitados en los 3 periodos
        # parcializacion, reconstruccion y consolidacion del hijo 1
        hijo1, rec_demand_cr_full_h1, rec_seg2_demand_cl_full, hijo1_I, hijo1_Q, hijo1_costo_f1, hijo1_costo_f2 = reconstruction_1(n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_p, capacidad_cr, capacidad_vehiculos_p, demandas_cl_cross_seg1, seg2_cr_habs_h1, seg2_vh1_habs, inventario, seg1_escalon, seg1_rutas, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano)
        # elementos que se heredan al hijo 1 del padre 2 y padre 1
        # en el hijo 2 se parcializa el segmento 1
        # elementos que se heredan al hijo 1 del padre 2 y padre 1
        seg2_cr_habs_h2 = extract_c(padre2, n_periodos, 1)  # centros regionales habilitados en el primer escalon del padre 2
        vh1_habs_h2 = extract_vh(padre2, n_periodos, 1)  # vehiculos de primer lvl habilitados en los 3 periodos del padre 2
        # seg2_cl_habs_h2 = extract_c(padre2, n_periodos, 0)  # centros locales habilitados en el primer escalon del padre 2
        vh2_habs_h2 = extract_vh(padre1, n_periodos, 3)   # vehiculos de segundo lvl habilitados en los 3 periodos del padre 1
        # pacializacion, reconstruccion y consolidacion del hijo 2
        hijo2, rec_seg2_demand_cr_full_h2, rec_seg1_demand_cl_full_h2, hijo2_I, hijo2_Q, hijo2_costo_f1, hijo2_costo_f2 = reconstruction_2(n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cl, capacidad_cr, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demandas_cl_cross_seg1, vh1_habs_h2, seg2_cr_habs_h2, vh2_habs_h2, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano)
        hijos.append(hijo1)
        hijos.append(hijo2)
        demand_cr_hijos.append(rec_demand_cr_full_h1)
        demand_cr_hijos.append(rec_seg2_demand_cr_full_h2)
        demand_cl_hijos.append(rec_seg2_demand_cl_full)
        demand_cl_hijos.append(rec_seg1_demand_cl_full_h2)
        I_hijos.append(hijo1_I)
        I_hijos.append(hijo2_I)
        Q_hijos.append(hijo1_Q)
        Q_hijos.append(hijo2_Q)
        f1_hijos.append(hijo1_costo_f1)
        f1_hijos.append(hijo2_costo_f1)
        f2_hijos.append(hijo2_costo_f1)
        f2_hijos.append(hijo2_costo_f2)

    return p_crossed, hijos, demand_cr_hijos, demand_cl_hijos, Q_hijos, I_hijos, f1_hijos, f2_hijos


def mutation(hijos, demandas_hijos, n_centrosregionales, capacidad_cr, n_periodos, n_productos, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, n_centroslocales, costo_vehiculos_s, costo_vehiculos_p, costo_humano, Q_hijos, I_hijos, f1_hijos, f2_hijos):
    prob_mut = 0.1
    h_muts = int(len(hijos) * prob_mut)
    idx_hijos = np.array(range(len(hijos)))
    muted = []
    for _ in range(h_muts):
        # para el proceso de mutacion se mutaran las asignaciones de primer nivel, es decir re asignaremos un centro regional
        # las demandas de los n productos en los n periodos de tiempo para cada centro se encuentran en rec_demand_cr_full_h1
        # se extraen los centros regionales habilitados en cada periodo de tiempo
        idx_hijo_copy = np.random.choice(idx_hijos)
        if idx_hijo_copy not in muted:
            muted.append(idx_hijo_copy)
            idx_hijos = np.delete(idx_hijos, np.where(idx_hijos == idx_hijo_copy))
        else:
            idx_hijo_copy = np.random.choice(idx_hijos)
        hijo_copy = hijos[idx_hijo_copy]
        demand_hijo_copy = np.copy(demandas_hijos[idx_hijo_copy])
        centros_dem = []
        for demanda in demand_hijo_copy:
            if demanda.keys() not in centros_dem:
                centros_dem.append(list(demanda.keys()))
        centros_habs = []
        for centros in centros_dem:
            for centro in centros:
                if centro not in centros_habs:
                    centros_habs.append(centro)
        # se hace una lista con los centros regionales no habilitados en cada periodo de tiempo
        centrosregionales = np.array(range(1, n_centrosregionales+1))
        for centro in centros_habs:
            centrosregionales = np.delete(centrosregionales, np.where(centrosregionales == int(centro)))
        # se selecciona aleatoriamente el centro regional que sera mutado
        cr_mut = np.random.choice(centros_habs)
        centros_copy = np.copy(centrosregionales)
        periodos_habs = []
        for period, demanda_periodo in enumerate(demand_hijo_copy):
            if cr_mut in demanda_periodo.keys():
                periodos_habs.append(period)
        # extraemos las demandas del centro regional a mutar en cada periodo de tiempo
        demandas_mut = []
        for p in periodos_habs:
            demandas_mut.append(demand_hijo_copy[p][int(cr_mut)])
        aceptado = 0
        cont = 0
        cr_remp = np.random.choice(centros_copy)
        while aceptado != 1:
            for p in range(len(periodos_habs)):
                resta = capacidad_cr[cr_remp-1, :] - demandas_mut[p]
                resvec = np.array([binarize(x) for x in resta])
                if resvec.all():
                    cont += 1
                    if cont == len(periodos_habs):
                        aceptado = 1
                    else:
                        continue
                else:
                    cont = 0
                    break
            if len(centros_copy) > 1 and aceptado != 1:
                centros_copy = np.delete(centros_copy, np.where(centros_copy == cr_remp))
                cr_remp = np.random.choice(centros_copy)
            elif len(centros_copy) == 1 and len(centros_habs) > 1 and aceptado != 1:
                centros_copy = np.copy(centrosregionales)
                centros_habs = np.delete(centros_habs, np.where(centros_habs == cr_mut))
                cr_mut = np.random.choice(centros_habs)
            elif len(centros_copy) == 1 and len(centros_habs) == 1 and aceptado != 1:
                cr_mut = 0
                cr_remp = 0
                break
                # return cr_mut, cr_remp
            elif aceptado == 1:
                continue
        # modificaciones al hijo
        if cr_mut != 0:
            asig_primer_lvl = hijo_copy[0]
            rutas_primer_lvl = hijo_copy[1]
            # modificacion del escalon
            for asig in asig_primer_lvl:
                asig[1] = np.where(asig[1] != cr_mut, asig[1], cr_remp)
            # modificacion de las rutas
            for ruta in rutas_primer_lvl:
                if ruta != -1 and ruta[0] == cr_mut:
                    ruta[0] = cr_remp
                else:
                    continue
            # modificacion a las demandas
            for demanda in demand_hijo_copy:
                if cr_mut in demanda.keys():
                    demanda[cr_remp] = demanda.pop(cr_mut)
                else:
                    continue
            # reconstruccion de las demas estructuras
            valoresQ, valoresI = fun_inventario(demand_hijo_copy, n_periodos, n_productos, n_centrosregionales, capacidad_cr, inventario)
            # costos f1
            cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1 = fitness_f1(n_periodos, n_productos, hijo_copy[2], costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, valoresQ, valoresI, hijo_copy[3], hijo_copy[1], costo_rutas_s, costo_rutas_p, n_centroslocales, n_centrosregionales, costo_vehiculos_s, costo_vehiculos_p)
            o = 10 ** -3
            costprod = costprod * o
            costtrans = costtrans * o
            costinv = costinv * o
            costo_f1 = np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1])
            # costos f2
            cost_sufr_hum = fitness_f2(hijo_copy[3], n_periodos, costo_humano, n_centroslocales)
            costo_f2 = -cost_sufr_hum
            hijo_copy[0] = asig_primer_lvl
            hijo_copy[1] = rutas_primer_lvl
            hijos[idx_hijo_copy] = hijo_copy
            demandas_hijos[idx_hijo_copy] = demand_hijo_copy
            Q_hijos[idx_hijo_copy] = valoresQ
            I_hijos[idx_hijo_copy] = valoresI
            f1_hijos[idx_hijo_copy] = costo_f1
            f2_hijos[idx_hijo_copy] = costo_f2

        else:
            ""
            # print("no se pudo")
            # return hijos, demandas_hijos, Q_hijos, I_hijos, fit_hijos

    return hijos, demandas_hijos, Q_hijos, I_hijos, f1_hijos, f2_hijos


def reduction(n_poblacion, frentes_dict, distancias_t):
    idx_poblation = []
    revised = []
    for frente, ind in frentes_dict.items():
        if len(idx_poblation) < n_poblacion:
            idx_poblation += ind
            revised.append(frente)
        else:
            break

    if len(idx_poblation) == n_poblacion:
        ""
    elif len(idx_poblation) < n_poblacion:
        rest = n_poblacion - len(idx_poblation)
        for frente, ind in frentes_dict.items():
            if len(idx_poblation) == n_poblacion:
                break
            else:
                if frente not in revised:
                    if rest == 2:
                        idx_poblation.append(ind[0])
                        idx_poblation.append(ind[-1])
                    elif rest < 2:
                        idx_poblation.append(ind[0])
                    elif rest > 2:
                        distancias = []
                        for i_temp in ind:
                            distancias.append(distancias_t[i_temp])
                        distancias_ord = sorted(distancias.items(), key=operator.itemgetter(1), reverse=True)
                        while rest > 0:
                            idx_poblation.append(distancias_ord.pop(0)[0])
                            rest -= 1
    elif len(idx_poblation) > n_poblacion:
        r = len(idx_poblation) - n_poblacion
        while r > 0:
            idx_poblation.pop()
            r -= 1

    return idx_poblation
