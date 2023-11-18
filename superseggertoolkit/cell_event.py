from enum import Enum
from networkx import Graph
from cell import Cell



# more strict cellEvent idea 
# this is used to label lineage 
class CellEvent(Enum):
    # this are event reprsented by edge 
    SPLIT = "split"
    MERGE = "merge"
    # this are event represented on node(cell)
    BIRTH = "birth"
    DIE = "die"



# more general cellEvent idea 
# this is used to label image/video 
class CellType(Enum):
    BIRTH = CellEvent.BIRTH
    DIE = CellEvent.DIE
    # split and merge is actually edge behavior
    # so need label both side of edge
    SPLIT = CellEvent.SPLIT
    MERGE = CellEvent.MERGE
    SPLITED = "splited"
    MERGED = "merged"
    #
    REGULAR = "regular"
    UNKOWN = "unkown"

    
# define the type/event of a cell
class CellDefine:
    def __init__(self, G:Graph, cell: Cell):
        self.cell = cell
        outgoing = len(list(G.successors(cell)))
        incoming = len(list(G.predecessors(cell)))

        # notice no more logic constained on split and merge 
        # Split and merge not conflict with each other, this dependednt on which direction of time you look for 
        # Split/merge not conflict with birth/die also
        self.split = (outgoing > 1)
        self.merge = (incoming > 1)

        self.birth = (incoming == 0) and (outgoing > 0)
        self.die = (outgoing == 0) and (incoming > 0)

        self.stary =  (incoming == 0) and (outgoing == 0) 

        self.regular = (outgoing == 1) and (incoming == 1)

    def __str__(self):
        return f"{self.cell}: split:{self.split}, merge:{self.merge}, birth:{self.birth}, die:{self.die}, regular:{self.regular}"
    
