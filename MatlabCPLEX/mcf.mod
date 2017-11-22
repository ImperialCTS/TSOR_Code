/*********************************************
 * OPL 12.6.1.0 Model
 * Author: jje11
 * Creation Date: 17 Nov 2017 at 16:25:05
 *********************************************/
tuple link {
	int orig;
	int dest;
	float cost;
	float capacity;
};

int 		NumNodes =...;
range  		Nodes = 1.. NumNodes;
{link} 		Links = ...;
int 		Demand[Nodes] = ...;
dvar float+ x[Links];
dvar float+ u[Nodes];

minimize sum (a in Links) a.cost * x[a] + sum(b in Nodes) abs(u[b]) * 100;
subject to {
	forall(a in Links) 
		0 <= x[a] <= a.capacity;
			
	forall(b in Nodes)
	  sum(a in Links:a.orig==b)x[a] -
	  sum(a in Links:a.dest==b)x[a] <= Demand[b] + u[b];
}

execute {

	var nFile = new IloOplOutputFile("Nodes.csv")
	var xFile = new IloOplOutputFile("Links.csv")
	var index = 0;
	
	for (var l in thisOplModel.Links)
	{
		index = index + 1;	
	    xFile.writeln("Link " + index + "," + thisOplModel.x[l]);
	}
	
	index = 0;
	for (var n in thisOplModel.Nodes)
	{
		index = index + 1;	
	    nFile.writeln("Node " + index + "," + thisOplModel.u[n]);
	}
	
	var rFile = new IloOplOutputFile("Costs.csv")
	var cost = cplex.getObjValue();
	rFile.writeln("Cost:," + cost);
	rFile.close();
	xFile.close();
	nFile.close();
}