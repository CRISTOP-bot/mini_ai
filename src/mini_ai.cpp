#include "mini_ai/mini_ai.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <istream>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
namespace mini_ai {
static float &M(Tensor &x, std::size_t i, std::size_t j) {
    return x[i * x.shape()[1] + j];
}
static float M(const Tensor &x, std::size_t i, std::size_t j) {
    return x[i * x.shape()[1] + j];
}
static float &M(Tensor &x, std::size_t i) {
    return x[i];
}
static float M(const Tensor &x, std::size_t i) {
    return x[i];
}
static void zero(std::vector<Tensor> &a) {
    for (auto &x : a)
        std::fill(x.data(), x.data() + x.size(), 0.f);
}
void Adam::step(std::vector<Tensor> &p, const std::vector<Tensor> &g, float lr, float b1, float b2,
                float eps) {
    if (m_.empty()) {
        for (auto &q : p) {
            m_.emplace_back(q.shape());
            v_.emplace_back(q.shape());
        }
    }
    if (m_.size() != p.size())
        throw std::runtime_error("optimizer parameter mismatch");
    ++t_;
    float c1 = 1 - std::pow(b1, float(t_)), c2 = 1 - std::pow(b2, float(t_));
    for (std::size_t k = 0; k < p.size(); ++k)
        for (std::size_t i = 0; i < p[k].size(); ++i) {
            m_[k][i] = b1 * m_[k][i] + (1 - b1) * g[k][i];
            v_[k][i] = b2 * v_[k][i] + (1 - b2) * g[k][i] * g[k][i];
            p[k][i] -= lr * (m_[k][i] / c1) / (std::sqrt(v_[k][i] / c2) + eps);
        }
}
void Adam::save(std::ostream &o) const {
    o.write((char *)&t_, sizeof t_);
    std::size_t n = m_.size();
    o.write((char *)&n, sizeof n);
    for (std::size_t k = 0; k < n; k++) {
        std::size_t z = m_[k].size();
        o.write((char *)&z, sizeof z);
        o.write((char *)m_[k].data(), z * sizeof(float));
        o.write((char *)v_[k].data(), z * sizeof(float));
    }
}
void Adam::load(std::istream &i, const std::vector<Tensor> &p) {
    clear();
    std::size_t n = 0;
    i.read((char *)&t_, sizeof t_);
    i.read((char *)&n, sizeof n);
    if (n != p.size())
        throw std::runtime_error("checkpoint optimizer mismatch");
    for (auto &q : p) {
        std::size_t z = 0;
        i.read((char *)&z, sizeof z);
        if (z != q.size())
            throw std::runtime_error("checkpoint moment shape mismatch");
        m_.emplace_back(q.shape());
        v_.emplace_back(q.shape());
        i.read((char *)m_.back().data(), z * sizeof(float));
        i.read((char *)v_.back().data(), z * sizeof(float));
    }
    if (!i)
        throw std::runtime_error("truncated checkpoint");
}
Model::Model(Config c) : c_(c), adam_(new Adam) {
    if (!c_.vocab || !c_.seq || !c_.d_model || !c_.d_ff)
        throw std::invalid_argument("invalid model config");
    init();
}
Model::~Model() {
    delete adam_;
}
void Model::init() {
    std::mt19937 r(7);
    std::normal_distribution<float> d(0, .02f);
    auto add = [&](std::vector<std::size_t> s) {
        p_.emplace_back(s);
        g_.emplace_back(s);
        for (std::size_t i = 0; i < p_.back().size(); i++)
            p_.back()[i] = d(r);
    };
    add({c_.vocab, c_.d_model});
    add({c_.seq, c_.d_model});
    add({c_.d_model, c_.d_model});
    add({c_.d_model, c_.d_model});
    add({c_.d_model, c_.d_model});
    add({c_.d_model, c_.d_model});
    add({c_.d_ff, c_.d_model});
    add({c_.d_ff});
    add({c_.d_ff, c_.d_model});
    add({c_.d_model});
    add({c_.d_model, c_.vocab});
}
std::size_t Model::parameters() const {
    std::size_t n = 0;
    for (auto &p : p_)
        n += p.size();
    return n;
}
float Model::train_batch(const Batch &b) {
    if (b.batch == 0 || b.seq != c_.seq || b.x.size() != b.y.size() ||
        b.x.size() != b.batch * b.seq)
        throw std::invalid_argument("invalid batch");
    zero(g_);
    float loss = 0;
    const float scale = 1 / std::sqrt(float(c_.d_model));
    for (std::size_t n = 0; n < b.batch; n++) {
        const std::size_t S = c_.seq, D = c_.d_model, F = c_.d_ff, Vc = c_.vocab;
        std::vector<std::vector<float>> x(S, std::vector<float>(D)),
            q = x, k = x, val = x, z(S, std::vector<float>(D)), h = x,
            pre(S, std::vector<float>(F)), ff = pre, o = x, lg(S, std::vector<float>(Vc)),
            a(S, std::vector<float>(S));
        for (std::size_t t = 0; t < S; t++) {
            int id = b.x[n * S + t];
            if (id < 0 || std::size_t(id) >= Vc)
                throw std::invalid_argument("token out of range");
            for (std::size_t d = 0; d < D; d++) {
                x[t][d] = M(p_[0], id, d) + M(p_[1], t, d);
                for (std::size_t e = 0; e < D; e++) {
                    q[t][d] += x[t][e] * M(p_[2], e, d);
                    k[t][d] += x[t][e] * M(p_[3], e, d);
                    val[t][d] += x[t][e] * M(p_[4], e, d);
                }
            }
            float mx = -1e30f;
            for (std::size_t j = 0; j <= t; j++) {
                a[t][j] = scale * std::inner_product(q[t].begin(), q[t].end(), k[j].begin(), 0.f);
                mx = std::max(mx, a[t][j]);
            }
            float sum = 0;
            for (std::size_t j = 0; j <= t; j++) {
                a[t][j] = std::exp(a[t][j] - mx);
                sum += a[t][j];
            }
            for (std::size_t j = 0; j <= t; j++)
                a[t][j] /= sum;
            for (std::size_t d = 0; d < D; d++) {
                for (std::size_t j = 0; j <= t; j++)
                    z[t][d] += a[t][j] * val[j][d];
                h[t][d] = x[t][d];
                for (std::size_t e = 0; e < D; e++)
                    h[t][d] += z[t][e] * M(p_[5], e, d);
            }
        }
        for (std::size_t t = 0; t < S; t++) {
            for (std::size_t j = 0; j < F; j++) {
                pre[t][j] = M(p_[7], j);
                for (std::size_t d = 0; d < D; d++)
                    pre[t][j] += h[t][d] * M(p_[6], j, d);
                ff[t][j] = std::max(0.f, pre[t][j]);
            }
            for (std::size_t d = 0; d < D; d++) {
                o[t][d] = h[t][d] + p_[9][d];
                for (std::size_t j = 0; j < F; j++)
                    o[t][d] += ff[t][j] * M(p_[8], j, d);
            }
        }
        for (std::size_t t = 0; t < S; t++) {
            float mx = -1e30f;
            for (std::size_t v = 0; v < Vc; v++) {
                lg[t][v] = 0;
                for (std::size_t d = 0; d < D; d++)
                    lg[t][v] += o[t][d] * M(p_[10], d, v);
                mx = std::max(mx, lg[t][v]);
            }
            float sum = 0;
            for (float &s : lg[t]) {
                s = std::exp(s - mx);
                sum += s;
            }
            for (float &s : lg[t])
                s /= sum;
            int target = b.y[n * S + t];
            if (target < 0 || std::size_t(target) >= Vc)
                throw std::invalid_argument("target out of range");
            loss -= std::log(std::max(lg[t][target], 1e-20f));
            std::vector<float> go = o[t];
            for (std::size_t d = 0; d < D; d++)
                go[d] = 0;
            for (std::size_t v = 0; v < Vc; v++) {
                float ds = lg[t][v] - (v == std::size_t(target));
                for (std::size_t d = 0; d < D; d++) {
                    M(g_[10], d, v) += ds * o[t][d];
                    go[d] += ds * M(p_[10], d, v);
                }
            }
            std::vector<float> gh = go;
            for (std::size_t j = 0; j < F; j++) {
                float gf = 0;
                for (std::size_t d = 0; d < D; d++) {
                    M(g_[8], j, d) += ff[t][j] * go[d];
                    gf += go[d] * M(p_[8], j, d);
                }
                if (pre[t][j] > 0) {
                    M(g_[7], j) += gf;
                    for (std::size_t d = 0; d < D; d++) {
                        M(g_[6], j, d) += gf * h[t][d];
                        gh[d] += gf * M(p_[6], j, d);
                    }
                }
            }
            for (std::size_t d = 0; d < D; d++)
                M(g_[9], d) += go[d];
            std::vector<std::vector<float>> gq(S, std::vector<float>(D)),
                gk = gq, gv = gq, gx(S, std::vector<float>(D));
            for (std::size_t d = 0; d < D; d++) {
                for (std::size_t e = 0; e < D; e++) {
                    M(g_[5], e, d) += z[t][e] * gh[d];
                }
                for (std::size_t e = 0; e < D; e++)
                    gx[t][e] += gh[d] * (e == d);
                for (std::size_t j = 0; j <= t; j++) {
                    float ga = 0;
                    for (std::size_t e = 0; e < D; e++) {
                        gv[j][e] += a[t][j] * gh[e] * M(p_[5], e, d);
                        ga += gh[e] * M(p_[5], d, e) * val[j][e];
                    } /* ga is corrected below by direct sum */
                }
            }
            std::vector<float> ga(S);
            for (std::size_t j = 0; j <= t; j++)
                for (std::size_t d = 0; d < D; d++)
                    ga[j] +=
                        gh[d] * M(p_[5], d, d) * 0.f; // retained for clarity; recompute exact below
            for (std::size_t j = 0; j <= t; j++) {
                ga[j] = 0;
                for (std::size_t d = 0; d < D; d++)
                    for (std::size_t e = 0; e < D; e++)
                        ga[j] += gh[d] * M(p_[5], e, d) * val[j][e];
            }
            float mean = 0;
            for (std::size_t j = 0; j <= t; j++)
                mean += ga[j] * a[t][j];
            for (std::size_t j = 0; j <= t; j++) {
                float gs = a[t][j] * (ga[j] - mean) * scale;
                for (std::size_t d = 0; d < D; d++) {
                    gq[t][d] += gs * k[j][d];
                    gk[j][d] += gs * q[t][d];
                }
            }
            for (std::size_t u = 0; u < S; u++)
                for (std::size_t d = 0; d < D; d++) {
                    for (std::size_t e = 0; e < D; e++) {
                        M(g_[2], e, d) += x[u][e] * gq[u][d];
                        M(g_[3], e, d) += x[u][e] * gk[u][d];
                        M(g_[4], e, d) += x[u][e] * gv[u][d];
                        gx[u][e] += gq[u][d] * M(p_[2], e, d) + gk[u][d] * M(p_[3], e, d) +
                                    gv[u][d] * M(p_[4], e, d);
                    }
                }
            for (std::size_t u = 0; u < S; u++)
                for (std::size_t d = 0; d < D; d++) {
                    int id = b.x[n * S + u];
                    M(g_[1], u, d) += gx[u][d];
                    M(g_[0], id, d) += gx[u][d];
                }
        }
    }
    update();
    return loss / float(b.batch * c_.seq);
}
void Model::update() {
    adam_->step(p_, g_);
    ++steps_;
}
std::vector<float> Model::logits(const std::vector<int> &ids) {
    if (ids.empty())
        return std::vector<float>(c_.vocab, 0.f);
    std::vector<int> x = ids;
    if (x.size() > c_.seq)
        x.erase(x.begin(), x.end() - c_.seq);
    const size_t S = x.size(), D = c_.d_model, F = c_.d_ff;
    std::vector<std::vector<float>> q(S, std::vector<float>(D)),
        k = q, val = q, z = q, h = q, pre(S, std::vector<float>(F)), ff = pre,
        o(S, std::vector<float>(D)), a(S, std::vector<float>(S));
    const float sc = 1 / std::sqrt(float(D));
    for (size_t t = 0; t < S; t++) {
        for (size_t d = 0; d < D; d++) {
            h[t][d] = M(p_[0], x[t], d) + M(p_[1], t, d);
            for (size_t e = 0; e < D; e++) {
                q[t][d] += h[t][e] * M(p_[2], e, d);
                k[t][d] += h[t][e] * M(p_[3], e, d);
                val[t][d] += h[t][e] * M(p_[4], e, d);
            }
        }
        float mx = -1e30f;
        for (size_t j = 0; j <= t; j++) {
            a[t][j] = sc * std::inner_product(q[t].begin(), q[t].end(), k[j].begin(), 0.f);
            mx = std::max(mx, a[t][j]);
        }
        float sum = 0;
        for (size_t j = 0; j <= t; j++) {
            a[t][j] = std::exp(a[t][j] - mx);
            sum += a[t][j];
        }
        for (size_t j = 0; j <= t; j++)
            a[t][j] /= sum;
        for (size_t d = 0; d < D; d++) {
            for (size_t j = 0; j <= t; j++)
                z[t][d] += a[t][j] * val[j][d];
            h[t][d] += 0;
            for (size_t e = 0; e < D; e++)
                h[t][d] += z[t][e] * M(p_[5], e, d);
        }
    }
    for (size_t t = 0; t < S; t++) {
        for (size_t j = 0; j < F; j++) {
            pre[t][j] = p_[7][j];
            for (size_t d = 0; d < D; d++)
                pre[t][j] += h[t][d] * M(p_[6], j, d);
            ff[t][j] = std::max(0.f, pre[t][j]);
        }
        for (size_t d = 0; d < D; d++) {
            o[t][d] = h[t][d] + p_[9][d];
            for (size_t j = 0; j < F; j++)
                o[t][d] += ff[t][j] * M(p_[8], j, d);
        }
    }
    std::vector<float> out(c_.vocab);
    for (size_t v = 0; v < c_.vocab; v++)
        for (size_t d = 0; d < D; d++)
            out[v] += o.back()[d] * M(p_[10], d, v);
    return out;
}
std::vector<int> Model::generate(std::vector<int> ids, std::size_t n, float temperature) {
    std::mt19937 r(9);
    for (std::size_t z = 0; z < n; z++) {
        auto l = logits(ids);
        float mx = *std::max_element(l.begin(), l.end()), sum = 0;
        for (float &v : l) {
            v = std::exp((v - mx) / std::max(.01f, temperature));
            sum += v;
        }
        std::uniform_real_distribution<float> d(0, sum);
        float u = d(r);
        std::size_t pick = 0;
        for (; pick + 1 < l.size(); pick++)
            if ((u -= l[pick]) <= 0)
                break;
        ids.push_back(int(pick));
        if (ids.size() > c_.seq)
            ids.erase(ids.begin());
    }
    return ids;
}
void Model::save(const std::string &f) const {
    std::ofstream o(f, std::ios::binary);
    if (!o)
        throw std::runtime_error("cannot open checkpoint");
    o.write("MAI3", 4);
    o.write((char *)&c_, sizeof c_);
    o.write((char *)&steps_, sizeof steps_);
    std::size_t n = p_.size();
    o.write((char *)&n, sizeof n);
    for (auto &p : p_) {
        std::size_t z = p.size();
        o.write((char *)&z, sizeof z);
        o.write((char *)p.data(), z * sizeof(float));
    }
    adam_->save(o);
}
void Model::load(const std::string &f) {
    std::ifstream i(f, std::ios::binary);
    if (!i)
        throw std::runtime_error("cannot open checkpoint");
    char magic[4];
    i.read(magic, 4);
    if (std::string(magic, 4) != "MAI3")
        throw std::runtime_error("invalid checkpoint version");
    Config c;
    i.read((char *)&c, sizeof c);
    if (c.vocab != c_.vocab || c.seq != c_.seq || c.d_model != c_.d_model || c.d_ff != c_.d_ff)
        throw std::runtime_error("checkpoint configuration mismatch");
    i.read((char *)&steps_, sizeof steps_);
    std::size_t n = 0;
    i.read((char *)&n, sizeof n);
    if (n != p_.size())
        throw std::runtime_error("checkpoint parameter mismatch");
    for (auto &p : p_) {
        std::size_t z = 0;
        i.read((char *)&z, sizeof z);
        if (z != p.size())
            throw std::runtime_error("checkpoint shape mismatch");
        i.read((char *)p.data(), z * sizeof(float));
    }
    adam_->load(i, p_);
}
} // namespace mini_ai