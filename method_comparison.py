#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:49:24 2020

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt

method = ['SVD', 'Biased SVD', 'implicit.ALS', 'surprise.NMF']


svd = [0.18768965517291686, .6120989226407433]
biased_svd = [.15415727, .56978107]
implicit = [4.637098841096637, 4.964006857458365]
surprise = [0.5049**2, 0.908**2]

X = np.array([1,1.7])
w = 0.1

# plt.figure()
# plt.bar(X - w, svd, color = 'b', width = 0.1, label=method[0])
# plt.bar(X, biased_svd, color = 'r', width = 0.1, label=method[1])
# plt.bar(X + w , implicit, color = 'g', width = 0.1, label=method[2])
# plt.bar(X + 2*w, surprise, color = 'k', width = 0.1, label=method[3])
# plt.xlim([0.7,2.1])
# plt.title('Mean Squared Errors of Different Matrix Factorization Methods')
# plt.xticks([1.05, 1.75], ('Train','Test'))
# plt.ylabel('Mean Squared Error')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.show()


f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

ax.bar(X - w, svd, color = 'b', width = 0.1, label=method[0])
ax.bar(X, biased_svd, color = 'r', width = 0.1, label=method[1])
ax.bar(X + w , implicit, color = 'g', width = 0.1, label=method[2])
ax.bar(X + 2*w, surprise, color = 'k', width = 0.1, label=method[3])
ax2.bar(X - w, svd, color = 'b', width = 0.1, label=method[0])
ax2.bar(X, biased_svd, color = 'r', width = 0.1, label=method[1])
ax2.bar(X + w , implicit, color = 'g', width = 0.1, label=method[2])
ax2.bar(X + 2*w, surprise, color = 'k', width = 0.1, label=method[3])

ax.set_ylim(4.4, 5.25)  # outliers only
ax2.set_ylim(0, 1.1)  # most of the data
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax2.set_xticks([1.05, 1.75])
ax2.set_xticklabels(['Train','Test'])
# ax2.xaxis.ticks([1.05, 1.75], ('Train','Test'))

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.title.set_text('MSE of Different Matrix Factorization Methods')
plt.xticks([1.05, 1.75], ('Train','Test'))
# plt.ylabel('Mean Squared Error')
lgd = plt.legend(bbox_to_anchor=(1, 2))

# plt.show()
plt.savefig('../plots/model_comparison.png', 
            bbox_extra_artists=(lgd,), bbox_inches='tight')
