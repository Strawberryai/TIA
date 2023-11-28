def puntos_comida(foodList):
    def get_lista_sin_elm(elm, lista):  
        lista_nueva = []
        for x in lista:
            if x != elm:
                lista_nueva.append(x)
        return lista_nueva
    def get_min(actual, esquinas):
        esq_min = None
        dist = None
        for e in esquinas:
            act_dist = util.manhattanDistance(actual, e)
            if dist is None:
                esq_min = e
                dist = act_dist
            elif act_dist < dist:
                esq_min = e
                dist = act_dist
        return dist, esq_min
    def get_coste_circuito(actual, esquinas_por_cal):
        coste = 0
        while len(esquinas_por_cal) > 0:
            dist, esquina = get_min(actual, esquinas_por_cal)
            esquinas_por_cal = get_lista_sin_elm(esquina, esquinas_por_cal)
            coste += dist
            actual = esquina
        return coste

    dists = []
    for food in foodList:
        dist = util.manhattanDistance(position, food)
        dist += get_coste_circuito(food, get_lista_sin_elm(food, foodList))
        dists.append(dist)
    if len(dists) == 0:
        return 0
    return min(dists)