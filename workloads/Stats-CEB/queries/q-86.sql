SELECT COUNT(*)
FROM comments AS c,
  votes AS v,
  users AS u,
  posts AS p
WHERE c.PostId = p.Id
  AND u.Id = c.UserId
  AND v.PostId = p.Id
  AND c.Score = 0
  AND u.Views >= 0
  AND u.Views <= 74;
