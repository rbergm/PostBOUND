SELECT COUNT(*)
FROM votes AS v, posts AS p, users AS u
WHERE v.UserId = u.Id
  AND p.OwnerUserId = u.Id
  AND p.CommentCount >= 0
  AND u.CreationDate >= CAST('2010-12-15 06:00:24' AS timestamp);
