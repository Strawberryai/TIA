def get_lista_sin_elm(elm, lista):
        lista_nueva = []
        for x in lista:
            if x != elm:
                lista_nueva.append(x)
        return lista_nueva
    
lista = []
print(min(lista))