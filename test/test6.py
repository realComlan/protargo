def random_generate_show(self):
                random_graph = nx.gnp_random_graph(30, 0.5, directed=True)
                G = nx.DiGraph([(u,v) for (u,v) in random_graph.edges() if u>v])
                pos = nx.spring_layout(G, seed=3068)  # Seed layout for reproducibility
                for layer, nodes in enumerate(nx.topological_generations(G)):
                        # `multipartite_layout` expects the layer as a node attribute, so add the
                        # numeric layer value as a node attribute
                        for node in nodes:
                                G.nodes[node]["layer"] = layer

                                # Compute the multipartite_layout using the "layer" node attribute
                                pos = nx.multipartite_layout(G, subset_key="layer")

                                fig, ax = plt.subplots()
                                nx.draw_networkx(G, pos=pos, ax=ax)
                                ax.set_title("DAG layout in topological order")
                                fig.tight_layout()
                                plt.show()

