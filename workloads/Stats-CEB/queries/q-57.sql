SELECT COUNT(*)
FROM tags AS t,
  posts AS p,
  users AS u,
  votes AS v,
  badges AS b
WHERE p.Id = t.ExcerptPostId
  AND u.Id = v.UserId
  AND u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND u.Views >= 0
  AND u.Views <= 515
  AND u.UpVotes >= 0
  AND u.CreationDate <= CAST('2014-09-07 13:46:41' AS timestamp)
  AND v.BountyAmount >= 0
  AND v.BountyAmount <= 200
  AND b.Date <= CAST('2014-09-12 12:56:22' AS timestamp);
