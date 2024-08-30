SELECT COUNT(*)
FROM tags AS t,
  posts AS p,
  users AS u,
  votes AS v,
  badges AS b
WHERE u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = v.UserId
  AND p.Id = t.ExcerptPostId
  AND p.CommentCount >= 0
  AND p.CommentCount <= 13
  AND u.Reputation >= 1
  AND b.Date <= CAST('2014-09-06 17:33:22' AS timestamp);
