    def _apply(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        dt = group["compute_dtype"]
        W = p.detach().to(dt)
        G = p.grad.detach().to(dt)
        basis_in, basis_out = state["bases"]

        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, basis_in)
        X_out = natural_gradient(G_out, basis_out)

        A = state["cov"].matrix.to(device=p.device, dtype=dt)
        eye_out = torch.eye(W.shape[0], dtype=dt, device=W.device)
        quad = (G_in * X_in).sum() + (G_out * X_out).sum()
        curv = (X_in * fisher_apply(A, W.T @ W, X_in)).sum() + (
            X_out * fisher_apply(eye_out, W @ A @ W.T, X_out)
        ).sum()
        alpha = trust_region_alpha(quad, curv, group["alpha_max"])
        state["alpha"] = float(alpha)

        c = group["lr"] * float(alpha)
        # recorded for diagnostics: dropping Pion's RMS scaling removed what
        # pinned the rotation angle, so whether it stays bounded is measured
        # rather than assumed
        state["angle"] = c * float(
            torch.maximum(
                torch.linalg.matrix_norm(X_in, 2), torch.linalg.matrix_norm(X_out, 2)
            )
        )
        if group["alternate"]:
            W = W @ cayley(X_in, c) if state["step"] % 2 else cayley(X_out, c) @ W
        else:
            W = cayley(X_out, c) @ W @ cayley(X_in, c)

        p.copy_(W.to(p.dtype))
        state["step"] += 1
        state["since_refactor"] += 1
