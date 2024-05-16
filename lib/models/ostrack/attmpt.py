##########################  vanilla refocus  ####################
    # def forward_features_refocus(self, z, x, return_attention=False, ffd_output_layer=None):
    #     # first feedforward
    #     out, aux_dict = self.forward_features(z, x, )
    #     # feature selection and feedback
    #     cos_sim = F.normalize(out, dim=-1) @ F.normalize(self.refocus_td_task_embed[None, ..., None], dim=1)  # B, N, 1
    #     mask = cos_sim.clamp(0, 1)
    #     out = out * mask
    #     out = out @ self.refocus_td_transform
    #
    #     td = []
    #     for depth in range(len(self.refocus_decoders) - 1, -1, -1):
    #         opt = self.refocus_decoders[depth](out)
    #         td = [opt] + td
    #     # second feedforward
    #     x, _ = self.forward_features(z, x, td=td)  # same input
    #     return x, aux_dict

#####################  refocus_query_rewrite_v2  ###########################
    # def forward_features_refocus(self, z, x, return_attention=False):
    #     # first feedforward
    #     out, aux_dict = self.forward_features(z, x, return_attention=return_attention)
    #     lens_z = self.pos_embed_z.shape[1]
    #     lens_x = self.pos_embed_x.shape[1]
    #     out_z = out[:, :lens_z]
    #     out_x = out[:, -lens_x:]
    #
    #     mean_z = torch.mean(out_z, dim=1, keepdim=True)    # B, 1, C
    #     index = slice(2, 6)
    #     agr_z = out_z.transpose(-1, -2).reshape(out_z.shape[0], -1, 8, 8)[:,:,index,index]
    #     agr_z = agr_z.flatten(2).transpose(-1, -2).mean(dim=1, keepdim=True)  # B, 1, C
    #
    #     # feature selection and feedback
    #     ## token selection
    #     spatial_specific_token = self.refocus_td_task_embed * agr_z
    #     cos_sim = F.normalize(out, dim=-1) @ F.normalize(spatial_specific_token.transpose(-1,-2), dim=1)  # B, N, 1
    #     mask = cos_sim.clamp(0, 1)
    #     out = out * mask
    #     ## channel selection
    #     channel_specific_token = mean_z @ self.refocus_td_transform
    #     out = out * channel_specific_token
    #     # out = torch.cat([out_z, out_x], dim=1)
    #
    #     td = []
    #     for depth in range(len(self.refocus_decoders) - 1, -1, -1):
    #         opt = self.refocus_decoders[depth](out)  # out,
    #         td = [opt] + td
    #     # second feedforward
    #     x, _ = self.forward_features(z, x, td=td, return_attention=return_attention)  # same input
    #     return x, aux_dict

    #####################  refocus_query_rewrite  ###########################
    # def forward_features_refocus(self, z, x):
    #     # first feedforward
    #     out, aux_dict = self.forward_features(z, x, )
    #     lens_z = self.pos_embed_z.shape[1]
    #     lens_x = self.pos_embed_x.shape[1]
    #     out_z = out[:, :lens_z]
    #     agr_z = torch.mean(out_z, dim=1, keepdim=True)    # B, 1, C
    #     max_z = torch.max(out_z, dim=1, keepdim=True)[0]  # B, 1, C
    #     # feature selection and feedback
    #     ## token selection
    #     spatial_specific_token = self.refocus_td_task_embed * agr_z
    #     cos_sim = F.normalize(out, dim=-1) @ F.normalize(spatial_specific_token.transpose(-1,-2), dim=1)  # B, N, 1
    #     mask = cos_sim.clamp(0, 1)
    #     out = out * mask
    #     ## channel selection
    #     channel_specific_token = max_z @ self.refocus_td_transform
    #     out = out * channel_specific_token
    #
    #     td = []
    #     for depth in range(len(self.refocus_decoders) - 1, -1, -1):
    #         out = self.refocus_decoders[depth](out)  # out,
    #         td = [out] + td
    #     # second feedforward
    #     x, _ = self.forward_features(z, x, td=td)  # same input
    #     return x, aux_dict