import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMDVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_depot: torch.Tensor #keeps track of current depot of route
    start_new_routes: torch.Tensor #keeps track of whether a new route has to be started in this step
    new_route_was_started: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    num_depots: int

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                num_depots=self.num_depots
            )
        return super(StateMDVRP, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']

        batch_size, n_loc, _ = loc.size()
        return StateMDVRP(
            coords=torch.cat((depot, loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + depot.size(1),
                    dtype=torch.uint8, device=loc.device
                )
                # if visited_dtype == torch.uint8 # unclear what this had achieved and thus also how to adapt; keep in mind in case it is important; maybe just unused piece of code
                # else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord = None,
            cur_depot = None,
            start_new_routes = torch.ones(batch_size, 1, device=loc.device, dtype=torch.bool),
            new_route_was_started = None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            num_depots = depot.size(1)
        )

    def get_final_cost(self):

        assert self.all_finished()
        #+ (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)
        return self.lengths.t()[0]

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths
        if self.cur_depot is not None:
            cur_depot = self.cur_depot.detach().clone()
        else:
            cur_depot = self.cur_depot
        if self.i>0:
            lengths[~self.start_new_routes] = self.lengths[~self.start_new_routes] + (cur_coord[~self.start_new_routes] - self.cur_coord[~self.start_new_routes]).norm(p=2, dim=-1)  # (batch_dim, 1)

            # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
            #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
            selected_demand = self.demand[self.ids, torch.clamp(prev_a - self.num_depots, 0, n_loc - 1)]
    
            # Increase capacity if depot is not visited, otherwise set to 0
            #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
            used_capacity = (self.used_capacity + selected_demand) * (prev_a >= self.num_depots).float()
            cur_depot[self.start_new_routes] = selected[self.start_new_routes]
            
        else:
            used_capacity = self.used_capacity
            cur_depot = selected

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)
        
        new_route_was_started = self.start_new_routes.detach().clone()
        start_new_routes = self.start_new_routes.detach().clone()
        start_new_routes[selected<self.num_depots] = True
        start_new_routes[self.start_new_routes] = False
        

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, cur_depot=cur_depot,
            start_new_routes=start_new_routes,
            new_route_was_started=new_route_was_started, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited[:,:,self.num_depots:].all() and (self.prev_a < self.num_depots).all()

    def get_finished(self):
        all_customers_visited = self.visited[:,:,self.num_depots:].sum(-1) == self.visited[:,:,self.num_depots:].size(-1)
        currently_in_depot = (self.prev_a < self.num_depots)
        return torch.logical_and(all_customers_visited, currently_in_depot)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        assert self.visited_.dtype == torch.uint8
        visited_loc = self.visited_[:, :, self.num_depots:]
        #else:
        #    visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap
        
        #init mask for depots
        mask_depot = torch.ones(self.visited_[:, :, 0:self.num_depots].size(), dtype=torch.bool)
        if self.i > 0: #in first step cur_depot is None
            #allow vehicle to return to its depot
            mask_depot.scatter_(2, self.cur_depot[:,None,:], False)
            
            #mask routes depots if the route was just started and thus a normal node should be chosen
            mask_depot[self.new_route_was_started] = True
            
            
        # Cannot visit the depot if just visited and still unserved nodes
        #mask_depot = (self.prev_a >= 0) & (self.prev_a < self.num_depots) & ((mask_loc == 0).int().sum(-1) > 0)
        
        #mask all problems that have to start a new route
        mask_loc[self.start_new_routes] = True
        mask_depot[self.start_new_routes] = False
        
        if self.i > 0: #in first step cur_depot is None
            #all finished_tours have to have their current depot available
            done_tours = self.get_finished()
            mask_depot[done_tours] = mask_depot[done_tours].scatter(1, self.cur_depot[done_tours.t()[0],:], False)
        
        
        return torch.cat((mask_depot, mask_loc), -1)

    def construct_solutions(self, actions):
        return actions
